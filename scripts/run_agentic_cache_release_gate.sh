#!/usr/bin/env bash
set -euo pipefail

# Exact-artifact, one-model-at-a-time cache lifecycle release authority.
# This wrapper is intentionally macOS/Apple-Silicon only. It continuously
# requires AC power, holds a caffeinate assertion, runs the same user-shaped
# lifecycle fixture for DeepSeek, Gemma, and Qwen, and binds all receipts to
# one packed crate and release binary.

EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA is required}
CRATE_SHA256=${CRATE_SHA256:?CRATE_SHA256 is required}
HF2Q_BIN=${HF2Q_BIN:?HF2Q_BIN is required}
EXPECTED_BINARY_SHA256=${EXPECTED_BINARY_SHA256:?EXPECTED_BINARY_SHA256 is required}
DEEPSEEK_MODEL=${DEEPSEEK_MODEL:?DEEPSEEK_MODEL is required}
GEMMA_MODEL=${GEMMA_MODEL:?GEMMA_MODEL is required}
QWEN_MODEL=${QWEN_MODEL:?QWEN_MODEL is required}
QWEN38_MODEL=${QWEN38_MODEL:?QWEN38_MODEL is required}
DEEPSEEK_MODEL_SHA256=${DEEPSEEK_MODEL_SHA256:?DEEPSEEK_MODEL_SHA256 is required}
GEMMA_MODEL_SHA256=${GEMMA_MODEL_SHA256:?GEMMA_MODEL_SHA256 is required}
QWEN_MODEL_SHA256=${QWEN_MODEL_SHA256:?QWEN_MODEL_SHA256 is required}
QWEN38_MODEL_SHA256=${QWEN38_MODEL_SHA256:?QWEN38_MODEL_SHA256 is required}
OUT_ROOT=${OUT_ROOT:-$(mktemp -d /var/tmp/hf2q-cache-release.XXXXXX)}
HF2Q_MODEL_VERIFICATION_CACHE_DIR=${HF2Q_MODEL_VERIFICATION_CACHE_DIR:-$OUT_ROOT/model-verification-cache}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"

if [[ ${MLX_NATIVE_SKIP_METALLIB+x} ]]; then
  echo "MLX_NATIVE_SKIP_METALLIB is forbidden for exact-artifact release builds" >&2
  exit 2
fi
if [[ ${HF2Q_THERMAL_SWIFT_BIN+x} ]]; then
  echo "HF2Q_THERMAL_SWIFT_BIN is reserved for isolated contract tests" >&2
  exit 2
fi
readonly HF2Q_THERMAL_SWIFT_BIN=/usr/bin/swift
[[ -x "$HF2Q_THERMAL_SWIFT_BIN" ]] || {
  echo "required system Swift probe is unavailable: $HF2Q_THERMAL_SWIFT_BIN" >&2
  exit 2
}
[[ -x /usr/bin/pgrep ]] || {
  echo "required model-runtime probe is unavailable: /usr/bin/pgrep" >&2
  exit 2
}
for command in awk caffeinate cargo cmp curl diff find jq lsof pmset rg sed shasum stat; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "EXPECTED_SHA must be a full Git SHA" >&2
  exit 2
}
[[ "$CRATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "CRATE_SHA256 must be a SHA-256 digest" >&2
  exit 2
}
[[ -x "$HF2Q_BIN" ]] || { echo "hf2q binary not executable: $HF2Q_BIN" >&2; exit 2; }
[[ "$EXPECTED_BINARY_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "EXPECTED_BINARY_SHA256 must be a lowercase 64-character digest" >&2
  exit 2
}
for model in "$DEEPSEEK_MODEL" "$GEMMA_MODEL" "$QWEN_MODEL" "$QWEN38_MODEL"; do
  [[ -f "$model" ]] || { echo "model not found: $model" >&2; exit 2; }
done
for digest in "$DEEPSEEK_MODEL_SHA256" "$GEMMA_MODEL_SHA256" \
  "$QWEN_MODEL_SHA256" "$QWEN38_MODEL_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "model SHA-256 values must be lowercase 64-character digests" >&2
    exit 2
  }
done

mkdir -p "$OUT_ROOT"
parent_pid=$$
server_pid=""
power_pid=""
caffeinate_pid=""
wave_harness_pid=""
cooperative_thermal_pid=""

require_ac() {
  local state
  state=$(pmset -g batt)
  printf 'sample_utc=%s\n%s\n' "$(date -u +%FT%TZ)" "$state" \
    >> "$OUT_ROOT/power.log"
  rg -q "Now drawing from 'AC Power'" <<<"$state" || {
    echo "cache lifecycle release gate requires continuous AC power" >&2
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
      printf '%s\n' "$assertions" > "$OUT_ROOT/power-assertions.current.txt"
      return 0
    fi
    sleep 1
  done
  echo "caffeinate process has no visible power assertion" >&2
  return 1
}

stop_server() {
  local deadline
  if [[ -z "$server_pid" ]] || ! kill -0 "$server_pid" 2>/dev/null; then
    server_pid=""
    return 0
  fi
  kill -INT "$server_pid" 2>/dev/null || true
  deadline=$((SECONDS + 180))
  while kill -0 "$server_pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 1
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    echo "server did not stop within 180 seconds" >&2
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
    return 1
  fi
  wait "$server_pid"
  server_pid=""
}

cleanup() {
  local cleanup_rc=0
  if [[ -n "$cooperative_thermal_pid" ]]; then
    [[ -z ${cooperative_thermal_stop:-} ]] || touch "$cooperative_thermal_stop"
    kill -TERM "$cooperative_thermal_pid" 2>/dev/null || true
    wait "$cooperative_thermal_pid" 2>/dev/null || true
    cooperative_thermal_pid=""
  fi
  if [[ -n "$wave_harness_pid" ]]; then
    kill -TERM "$wave_harness_pid" 2>/dev/null || true
    # Closing the server makes any in-flight curl return promptly before the
    # harness parent is reaped. A second stop_server below is then a no-op.
    stop_server || cleanup_rc=1
    wait "$wave_harness_pid" 2>/dev/null || true
    wave_harness_pid=""
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
  return "$cleanup_rc"
}
on_exit() {
  local original_rc=$?
  trap - EXIT
  if ! cleanup && (( original_rc == 0 )); then
    original_rc=1
  fi
  exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 1' INT TERM

require_ac
pmset -g assertions > "$OUT_ROOT/power-assertions.before.txt"
caffeinate -dimsu -w "$parent_pid" &
caffeinate_pid=$!
(
  while kill -0 "$parent_pid" 2>/dev/null; do
    if ! require_ac; then
      printf 'AC power lost at %s\n' "$(date -u +%FT%TZ)" > "$OUT_ROOT/power-failure.txt"
      kill -TERM "$parent_pid" 2>/dev/null || true
      exit 1
    fi
    sleep 5
  done
) &
power_pid=$!
ensure_guard_health

wait_ready() {
  local base_url=$1
  local log=$2
  local deadline=$((SECONDS + 360))
  while (( SECONDS < deadline )); do
    if curl --fail --silent --show-error --max-time 2 "$base_url/readyz" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "server exited before readiness" >&2
      sed -n '1,240p' "$log" >&2
      return 1
    fi
    sleep 1
  done
  echo "server did not become ready within 360 seconds" >&2
  sed -n '1,240p' "$log" >&2
  return 1
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

assert_exact_binary() {
  [[ -n "${binary_sha:-}" && -x "$HF2Q_BIN" ]] || {
    echo "release binary is missing or non-executable: $HF2Q_BIN" >&2
    return 1
  }
  "$script_dir/seal_release_binary.sh" --verify "$HF2Q_BIN" "$binary_sha" \
    >/dev/null
}

require_no_model_runtime() {
  local name
  for name in hf2q llama-server llama-cli; do
    if /usr/bin/pgrep -x "$name" >/dev/null 2>&1; then
      echo "calibrated wave requires no existing $name runtime" >&2
      /usr/bin/pgrep -flx "$name" >&2 || true
      return 1
    fi
  done
}

verify_sha256_sidecar() {
  local receipt=$1
  [[ -s "$receipt" ]] || {
    echo "receipt is missing or empty: $receipt" >&2
    return 1
  }
  [[ -s "$receipt.sha256" ]] || {
    echo "receipt checksum sidecar is missing or empty: $receipt.sha256" >&2
    return 1
  }
  shasum -a 256 -c "$receipt.sha256" >/dev/null
}

verify_model() {
  local family=$1
  local model=$2
  local expected=$3
  ensure_guard_health
  mkdir -p "$OUT_ROOT/$family"
  hf2q_release_prepare_model_verification "$model" "$expected" \
    "$OUT_ROOT/$family/model-verification.json" \
    "$HF2Q_MODEL_VERIFICATION_CACHE_DIR"
  printf '%s  %s\n' "$expected" "$model" > "$OUT_ROOT/$family/model.sha256"
}

current_dir=""
current_log=""
current_url=""

start_server() {
  local family=$1
  local phase=$2
  local launcher=$3
  local model=$4
  local port=$5
  local max_slots=$6
  local context_len=${7:-}
  local kv_budget=${8:-}
  local model_sha256 model_verification_receipt
  local -a launcher_env

  # Cargo parity tests later in this gate may relink package-local bin targets.
  # Every model must execute the immutable copy sealed outside Cargo's target.
  assert_exact_binary
  ensure_guard_health
  case "$family" in
    deepseek)
      model_sha256=$DEEPSEEK_MODEL_SHA256
      model_verification_receipt="$OUT_ROOT/deepseek/model-verification.json"
      ;;
    gemma)
      model_sha256=$GEMMA_MODEL_SHA256
      model_verification_receipt="$OUT_ROOT/gemma/model-verification.json"
      ;;
    qwen)
      model_sha256=$QWEN_MODEL_SHA256
      model_verification_receipt="$OUT_ROOT/qwen/model-verification.json"
      ;;
    *)
      echo "unknown release model family: $family" >&2
      return 1
      ;;
  esac
  hf2q_release_verify_model "$model" "$model_sha256" \
    "$model_verification_receipt"
  current_dir="$OUT_ROOT/$family/$phase"
  current_log="$current_dir/server.log"
  current_url="http://127.0.0.1:$port"
  mkdir -p "$current_dir"
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | rg -q .; then
    echo "release gate port already in use: $port" >&2
    return 1
  fi
  launcher_env=(MODEL="$model" PORT="$port" HF2Q_BIN="$HF2Q_BIN" MAX_SLOTS="$max_slots")
  [[ -z "$context_len" ]] || launcher_env+=(CONTEXT_LEN="$context_len")
  [[ -z "$kv_budget" ]] || launcher_env+=(KV_CACHE_BUDGET_BYTES="$kv_budget")
  if [[ "$family" == gemma ]]; then
    local disabled_mmproj="$current_dir/mmproj-disabled"
    [[ ! -e "$disabled_mmproj" ]] || return 1
    launcher_env+=(MMPROJ="$disabled_mmproj")
  fi
  env "${launcher_env[@]}" "$launcher" >"$current_log" 2>&1 &
  server_pid=$!
  wait_ready "$current_url" "$current_log"
}

finish_server_phase() {
  local ready_file="$current_dir/readyz.json"
  local ready_code
  ready_code=$(curl --silent --show-error --max-time 3 -o "$ready_file" \
    -w '%{http_code}' "$current_url/readyz")
  [[ "$ready_code" == 200 ]] || return 1
  stop_server
  if rg -n 'GPU Timeout|SubmissionsIgnored|engine_unhealthy|panicked at|fatal command.buffer' \
    "$current_log" >"$current_dir/fatal.log"; then
    cat "$current_dir/fatal.log" >&2
    return 1
  fi
  : >"$current_dir/fatal.log"
  sha256_file "$current_log" >"$current_dir/server.log.sha256"
  ensure_guard_health
}

run_lifecycle() {
  local family=$1
  local context_lines=$2
  local out="$OUT_ROOT/$family/lifecycle"
  mkdir -p "$out"
  BASE_URL="$current_url" \
  OUT_DIR="$out" \
  RUN_ID="release-${EXPECTED_SHA:0:12}-$family" \
  CONTEXT_LINES="$context_lines" \
  CURL_MAX_TIME_SECONDS=1800 \
  SEMANTIC_WAIT_SECONDS=300 \
    scripts/test_agentic_cache_lifecycle.sh \
      >"$out/stdout.log" 2>"$out/stderr.log"
  jq -e '.status == "pass"' "$out/summary.json" >/dev/null
  sha256_file "$out/summary.json" >"$out/summary.json.sha256"
}

binary_sha=$EXPECTED_BINARY_SHA256
assert_exact_binary
verify_model deepseek "$DEEPSEEK_MODEL" "$DEEPSEEK_MODEL_SHA256"
verify_model gemma "$GEMMA_MODEL" "$GEMMA_MODEL_SHA256"
verify_model qwen "$QWEN_MODEL" "$QWEN_MODEL_SHA256"
verify_model qwen38 "$QWEN38_MODEL" "$QWEN38_MODEL_SHA256"

mkdir -p "$OUT_ROOT/fixtures"
agentic_prompt_contract="$PWD/scripts/fixtures/deepseek4-agentic-prompt-contract-v2.json"
agentic_prompt_contract_sha=$(sha256_file "$agentic_prompt_contract")
agentic_fixture="$PWD/scripts/fixtures/deepseek4-agentic-repo-context.txt"
agentic_fixture_sha=$(jq -er '.repository_context.sha256' "$agentic_prompt_contract")
agentic_tool_result="$PWD/$(jq -er '.tool_result.path' "$agentic_prompt_contract")"
agentic_fixture_evidence="$OUT_ROOT/fixtures/deepseek4-agentic"
jq -e -f scripts/deepseek4_agentic_prompt_contract.jq \
  "$agentic_prompt_contract" >/dev/null
test "$(jq -er '.model.artifact_sha256' "$agentic_prompt_contract")" = \
  "$DEEPSEEK_MODEL_SHA256"
test "$(stat -f %z "$DEEPSEEK_MODEL")" = \
  "$(jq -er '.model.bytes' "$agentic_prompt_contract")"
for input in request_builder chat_template repository_context tool_result; do
  input_path=$(jq -er --arg input "$input" '.[$input].path' "$agentic_prompt_contract")
  test "$(sha256_file "$PWD/$input_path")" = \
    "$(jq -er --arg input "$input" '.[$input].sha256' "$agentic_prompt_contract")"
  test "$(stat -f %z "$PWD/$input_path")" = \
    "$(jq -er --arg input "$input" '.[$input].bytes' "$agentic_prompt_contract")"
done
test "$(jq -Rs 'length' "$agentic_fixture")" = \
  "$(jq -er '.repository_context.chars' "$agentic_prompt_contract")"
test "$(jq -Rs 'length' "$agentic_tool_result")" = \
  "$(jq -er '.tool_result.chars' "$agentic_prompt_contract")"
agentic_payload_sha=$(
  { jq -j '.tool_result.success_prefix' "$agentic_prompt_contract"; \
    cat "$agentic_tool_result"; } | shasum -a 256 | awk '{print $1}'
)
test "$agentic_payload_sha" = \
  "$(jq -er '.tool_result.combined_payload_sha256' "$agentic_prompt_contract")"

mkdir -p "$agentic_fixture_evidence"
cp "$agentic_prompt_contract" "$agentic_fixture_evidence/"
cp "$agentic_fixture" "$agentic_fixture_evidence/"
cp "$agentic_tool_result" "$agentic_fixture_evidence/"
for agent in 1 2 3 4; do
  request_path="$agentic_fixture_evidence/request-agent-$agent.json"
  provenance_path="$agentic_fixture_evidence/provenance-agent-$agent.json"
  HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$request_path" \
  AGENTIC_PROMPT_CONTRACT="$agentic_prompt_contract" \
  AGENTIC_PROMPT_CONTRACT_SHA256="$agentic_prompt_contract_sha" \
  AGENT_INDEX="$agent" \
  EXPECTED_PATH=/opt/hf2q-worktrees/full-context-slots/Cargo.toml \
  TOOL_RESULT_PATH="$agentic_tool_result" \
  AGENTIC_CONTEXT_FIXTURE="$agentic_fixture" \
  AGENTIC_CONTEXT_FIXTURE_SHA256="$agentic_fixture_sha" \
  RUN_ID="full-context-deepseek4-agent-$agent" \
  SENTINEL="HF2Q_DEEPSEEK4_AGENT_${agent}_OK" \
    bash scripts/test_deepseek4_agentic.sh
  HF2Q_DEEPSEEK4_GGUF="$DEEPSEEK_MODEL" \
  HF2Q_DEEPSEEK4_AGENTIC_REQUEST_JSON="$request_path" \
  HF2Q_DEEPSEEK4_AGENTIC_PROMPT_CONTRACT="$agentic_prompt_contract" \
  HF2Q_DEEPSEEK4_AGENTIC_CONTRACT_RECEIPT="$provenance_path" \
    cargo test --locked --bin hf2q \
      release_agentic_fixture_preserve_order_contract -- \
      --ignored --test-threads=1 \
      >"$agentic_fixture_evidence/provenance-agent-$agent.log" 2>&1
done
jq -s --arg contract_sha256 "$agentic_prompt_contract_sha" \
  --arg model_sha256 "$DEEPSEEK_MODEL_SHA256" '
  sort_by(.agent) as $agents
  | if length != 4 or ([.[].agent] != [1,2,3,4])
      or any(.[]; .status != "pass"
        or .prompt_contract_sha256 != $contract_sha256
        or .prompt_tokens != 6684
        or .legacy_key_sorted_prompt_tokens != 6685
        or .preserve_order_delta_proven != true)
    then error("invalid DeepSeek prompt provenance")
    else {
      schema_version:2,
      status:"pass",
      prompt_contract_sha256:$contract_sha256,
      model_sha256:$model_sha256,
      agents:$agents
    }
    end
' "$agentic_fixture_evidence"/provenance-agent-?.json \
  >"$agentic_fixture_evidence/prompt-provenance.json.tmp"
agentic_prompt_provenance="$agentic_fixture_evidence/prompt-provenance.json"
jq -e --slurpfile contract "$agentic_prompt_contract" \
  --arg prompt_contract_sha256 "$agentic_prompt_contract_sha" \
  --arg model_sha256 "$DEEPSEEK_MODEL_SHA256" \
  -f scripts/deepseek4_agentic_prompt_provenance.jq \
  "$agentic_prompt_provenance.tmp" >/dev/null
mv "$agentic_prompt_provenance.tmp" "$agentic_prompt_provenance"
agentic_prompt_provenance_sha=$(sha256_file "$agentic_prompt_provenance")

cooperative_dir="$OUT_ROOT/deepseek/cooperative-prefill"
cooperative_raw="$cooperative_dir/raw.json"
cooperative_log="$cooperative_dir/test.log"
cooperative_thermal_log="$cooperative_dir/thermal.log"
cooperative_thermal_stop="$cooperative_dir/thermal.stop"
cooperative_settle_log="$cooperative_dir/settle.log"
cooperative_build_log="$cooperative_dir/build.jsonl"
mkdir -p "$cooperative_dir"
cargo test --release --locked --bin hf2q --no-run --message-format=json \
  >"$cooperative_build_log"
cooperative_test_binary=$(jq -rs '
  [.[] | select(.reason == "compiler-artifact"
    and .profile.test == true
    and (.target.kind | index("bin")) != null
    and .target.name == "hf2q") | .executable]
  | map(select(type == "string" and length > 0)) | last // empty
' "$cooperative_build_log")
[[ -n "$cooperative_test_binary" && -x "$cooperative_test_binary" ]] || {
  echo "failed to resolve the prebuilt hf2q release test binary" >&2
  exit 1
}
require_no_model_runtime
thermal_wait_for_nominal "$cooperative_settle_log" \
  cooperative-prefill-settle 60 900 5
: >"$cooperative_thermal_log"
rm -f "$cooperative_thermal_stop"
thermal_sample "$cooperative_thermal_log" \
  cooperative-prefill-measurement-start
thermal_monitor_nominal "$cooperative_thermal_log" \
  cooperative-prefill-measurement "$cooperative_thermal_stop" 2 &
cooperative_thermal_pid=$!
cooperative_test_rc=0
env -i \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  TMPDIR="${TMPDIR:-/tmp}" \
  HF2Q_DEEPSEEK4_COHORT_BENCH_PAIRS=5 \
  HF2Q_DEEPSEEK4_GGUF="$DEEPSEEK_MODEL" \
  HF2Q_DEEPSEEK4_COHORT_RECEIPT="$cooperative_raw" \
  "$cooperative_test_binary" \
    official_artifact_cooperative_warm_prefill_is_exact_and_faster \
    --ignored --test-threads=1 --nocapture \
    >"$cooperative_log" 2>&1 || cooperative_test_rc=$?
touch "$cooperative_thermal_stop"
cooperative_thermal_rc=0
wait "$cooperative_thermal_pid" || cooperative_thermal_rc=$?
cooperative_thermal_pid=""
thermal_sample "$cooperative_thermal_log" \
  cooperative-prefill-measurement-end
test "$cooperative_test_rc" = 0
test "$cooperative_thermal_rc" = 0
thermal_validate_measurement_log "$cooperative_thermal_log" 5
cooperative_measurement_samples=$THERMAL_LOG_SAMPLES
cooperative_measurement_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
cooperative_non_nominal_samples=$THERMAL_LOG_NON_NOMINAL_SAMPLES
cooperative_measurement_gaps=$THERMAL_LOG_GAPS
thermal_validate_settle_log "$cooperative_settle_log" 60 8
cooperative_settle_samples=$THERMAL_LOG_SAMPLES
cooperative_settle_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
cooperative_settle_gaps=$THERMAL_LOG_GAPS
jq -e '
  .schema_version == 1 and .status == "pass"
  and .artifact_bytes == 107431343168 and .layers == 43
  and .prefix_rows == 148 and .prefix_mod_128 == 20 and .prefix_mod_4 == 0
  and [.parity_shapes[] | [.sequences,.rows_per_lane,.aggregate_rows]]
    == [[2,1024,2048],[3,640,1920],[4,512,2048]]
  and all(.parity_shapes[]; .exact_state_logits_decode == true)
  and .benchmark.sequences == 4 and .benchmark.rows_per_lane == 512
  and .benchmark.aggregate_rows == 2048 and .benchmark.pairs >= 5
  and .benchmark.order == "alternating"
  and (.benchmark.serial_ms | length) == .benchmark.pairs
  and (.benchmark.cohort_ms | length) == .benchmark.pairs
  and all(.benchmark.serial_ms[], .benchmark.cohort_ms[]; type == "number" and . > 0)
  and .benchmark.serial_median_ms > .benchmark.cohort_median_ms
  and .benchmark.speedup > 1
  and .benchmark.process_lifetime_peak_rss_bytes > 0
' "$cooperative_raw" >/dev/null
jq --arg source_sha "$EXPECTED_SHA" \
  --arg model_sha256 "$DEEPSEEK_MODEL_SHA256" \
  --arg mlx_native_version "0.10.12" \
  --arg raw_sha256 "$(sha256_file "$cooperative_raw")" \
  --arg test_log_sha256 "$(sha256_file "$cooperative_log")" \
  --arg measurement_log_sha256 "$(sha256_file "$cooperative_thermal_log")" \
  --arg settle_log_sha256 "$(sha256_file "$cooperative_settle_log")" \
  --argjson settle_seconds 60 \
  --argjson settle_samples "$cooperative_settle_samples" \
  --argjson settle_duration_seconds "$cooperative_settle_duration_seconds" \
  --argjson settle_sample_interval_seconds 5 \
  --argjson maximum_settle_sample_gap_seconds 8 \
  --argjson settle_telemetry_gaps "$cooperative_settle_gaps" \
  --argjson measurement_samples "$cooperative_measurement_samples" \
  --argjson measurement_duration_seconds "$cooperative_measurement_duration_seconds" \
  --argjson sample_interval_seconds 2 \
  --argjson maximum_sample_gap_seconds 5 \
  --argjson non_nominal_measurement_samples "$cooperative_non_nominal_samples" \
  --argjson telemetry_gaps "$cooperative_measurement_gaps" \
  '. + {source_sha:$source_sha,model_sha256:$model_sha256,
    mlx_native_version:$mlx_native_version,raw_sha256:$raw_sha256,
    test_log_sha256:$test_log_sha256,thermal_status:"nominal",
    measurement_log_sha256:$measurement_log_sha256,
    settle_log_sha256:$settle_log_sha256,settle_seconds:$settle_seconds,
    settle_samples:$settle_samples,
    settle_duration_seconds:$settle_duration_seconds,
    settle_sample_interval_seconds:$settle_sample_interval_seconds,
    maximum_settle_sample_gap_seconds:$maximum_settle_sample_gap_seconds,
    settle_telemetry_gaps:$settle_telemetry_gaps,
    measurement_samples:$measurement_samples,
    measurement_duration_seconds:$measurement_duration_seconds,
    sample_interval_seconds:$sample_interval_seconds,
    maximum_sample_gap_seconds:$maximum_sample_gap_seconds,
    non_nominal_measurement_samples:$non_nominal_measurement_samples,
    telemetry_gaps:$telemetry_gaps}' \
  "$cooperative_raw" >"$cooperative_dir/summary.json"
bash scripts/verify_deepseek4_cooperative_prefill_receipt.sh \
  "$cooperative_dir/summary.json" "$cooperative_raw" "$cooperative_log" \
  "$cooperative_thermal_log" "$cooperative_settle_log" \
  "$EXPECTED_SHA" "$DEEPSEEK_MODEL_SHA256"
sha256_file "$cooperative_dir/summary.json" \
  >"$cooperative_dir/summary.json.sha256"
ensure_guard_health

# The warm-prefill proof above and the exact decode proof exercise distinct
# transaction shapes. Reuse the already-built packed test binary and the
# already-verified model identity; do not rescan the 107 GB artifact.
require_no_model_runtime
bash scripts/run_deepseek4_decode_cohort_gate.sh \
  "$cooperative_test_binary" "$DEEPSEEK_MODEL" \
  "$OUT_ROOT/deepseek/decode-cohort" "$EXPECTED_SHA" \
  "$DEEPSEEK_MODEL_SHA256"
ensure_guard_health

HF2Q_QWEN36_WATCHDOG_FIXTURE_MODEL="$QWEN_MODEL" \
HF2Q_QWEN36_WATCHDOG_FIXTURE_OUTPUT="$OUT_ROOT/fixtures/public-347.json" \
HF2Q_QWEN36_WATCHDOG_SHORT_FIXTURE_OUTPUT="$OUT_ROOT/fixtures/public-short.json" \
  cargo test --locked --bin hf2q \
    public_347_tool_fixture_renders_to_exact_87972_tokens -- \
    --ignored --test-threads=1 >"$OUT_ROOT/fixtures/generator.log" 2>&1
test "$(sha256_file "$OUT_ROOT/fixtures/public-347.json")" = \
  "3558d4f4b251ed833ee7da1b037fa3f241a4309590d45930b525b690f543a31e"
test "$(sha256_file "$OUT_ROOT/fixtures/public-short.json")" = \
  "7aeddea35e6363c698ea0bcb4934b9f2cf1e0c48fb2045fa9db3272461e54004"

# Fail fast on prompt-render/tokenizer drift before paying the cost of loading
# the 100 GiB DeepSeek verifier. This request is byte-for-byte the first
# request used by the four-agent wave below.
deepseek_agentic_request="$OUT_ROOT/fixtures/deepseek-agentic-request.json"
HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$deepseek_agentic_request" \
RUN_ID=full-context-deepseek4-agent-1 \
SENTINEL=HF2Q_DEEPSEEK4_AGENT_1_OK \
EXPECTED_PATH=/opt/hf2q-worktrees/full-context-slots/Cargo.toml \
TOOL_RESULT_PATH="$PWD/Cargo.toml" \
AGENTIC_FIXTURE_ID=full-context-agentic-v1 EXPECTED_PROMPT_TOKENS=6684 \
AGENTIC_CONTEXT_FIXTURE="$agentic_fixture" \
AGENTIC_CONTEXT_FIXTURE_SHA256="$agentic_fixture_sha" \
  scripts/test_deepseek4_agentic.sh
HF2Q_DEEPSEEK4_GGUF="$DEEPSEEK_MODEL" \
HF2Q_DEEPSEEK4_AGENTIC_REQUEST_JSON="$deepseek_agentic_request" \
HF2Q_DEEPSEEK4_EXPECTED_PROMPT_TOKENS=6684 \
  cargo test --locked --bin hf2q \
    release_agentic_fixture_renders_to_expected_tokens -- \
    --ignored --test-threads=1 >"$OUT_ROOT/fixtures/deepseek-prompt-preflight.log" 2>&1

run_deepseek_wave() {
  local wave=$1
  local out="$OUT_ROOT/deepseek/full-context-$wave"
  local harness_rc=0
  local thermal_rc=0
  local thermal_dir="$out/thermal"
  local thermal_measurement_log="$thermal_dir/measurement.log"
  local thermal_settle_log="$thermal_dir/settle.log"
  local thermal_summary="$thermal_dir/summary.json"
  local cold_receipts_json
  local cooperative_prefill_transactions
  local decode_cohort_transactions
  local measurement_samples
  local measurement_duration_seconds
  local non_nominal_measurement_samples
  local settle_samples
  local settle_duration_seconds
  local settle_telemetry_gaps
  local telemetry_gaps
  local sample_interval_seconds=2
  local maximum_sample_gap_seconds=5
  local settle_sample_interval_seconds=5
  local maximum_settle_sample_gap_seconds=8
  mkdir -p "$out" "$thermal_dir"
  thermal_prepare_cold_receipt_dir "$out/agents"
  require_no_model_runtime
  thermal_wait_for_nominal "$thermal_settle_log" "deepseek-wave-${wave}-settle" \
    60 900 "$settle_sample_interval_seconds"
  start_server deepseek "process-wave-$wave" scripts/serve_deepseek4_opencode.sh \
    "$DEEPSEEK_MODEL" 18080 4 524288 8589934592
  : >"$thermal_measurement_log"
  thermal_sample "$thermal_measurement_log" \
    "deepseek-wave-${wave}-measurement-start"
  test "$THERMAL_STATE" = nominal
  # Preserve the prompt-visible historical path while feeding the immutable
  # bytes that actually lived there at the calibrated source revision.
  BASE_URL="$current_url" FAMILY=deepseek4 AGENTS=4 \
  WAVE_ID=default REQUIRE_COLD_FIRST=1 \
  EXPECTED_PATH=/opt/hf2q-worktrees/full-context-slots/Cargo.toml \
  TOOL_RESULT_PATH="$agentic_tool_result" \
  AGENTIC_PROMPT_CONTRACT="$agentic_prompt_contract" \
  AGENTIC_PROMPT_CONTRACT_SHA256="$agentic_prompt_contract_sha" \
  PROMPT_PROVENANCE_SHA256="$agentic_prompt_provenance_sha" \
  AGENTIC_CONTEXT_FIXTURE="$agentic_fixture" \
  AGENTIC_CONTEXT_FIXTURE_SHA256="$agentic_fixture_sha" \
  MAX_COLD_TTFT_MS=60000 MAX_COLD_RESPONSE_MS=60000 \
  MAX_CACHED_TTFT_MS=5000 MAX_CACHED_RESPONSE_MS=15000 \
  MAX_CACHED_SEMANTIC_MS=15000 MAX_TOOL_RESULT_RESPONSE_MS=35000 \
  CURL_CONNECT_TIMEOUT_SECONDS=5 CURL_MAX_TIME_SECONDS=90 \
  OUT_DIR="$out/agents" scripts/test_full_context_agent_slots.sh \
    >"$out/summary.json.tmp" 2>"$out/harness.stderr" &
  wave_harness_pid=$!
  set +e
  thermal_monitor_nominal_until_cold_receipts "$thermal_measurement_log" \
    "deepseek-wave-${wave}-measurement" "$out/agents" 4 \
    "$sample_interval_seconds" 120 "$wave_harness_pid"
  thermal_rc=$?
  if ((thermal_rc != 0)); then
    kill -TERM "$wave_harness_pid" 2>/dev/null || true
    stop_server || true
  fi
  wait "$wave_harness_pid"
  harness_rc=$?
  wave_harness_pid=""
  set -e
  ((thermal_rc == 0)) || {
    echo "DeepSeek wave $wave cold-cohort thermal monitor failed" >&2
    return 1
  }
  ((harness_rc == 0)) || return "$harness_rc"
  thermal_validate_measurement_log "$thermal_measurement_log" \
    "$maximum_sample_gap_seconds"
  measurement_samples=$THERMAL_LOG_SAMPLES
  measurement_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
  non_nominal_measurement_samples=$THERMAL_LOG_NON_NOMINAL_SAMPLES
  telemetry_gaps=$THERMAL_LOG_GAPS
  thermal_validate_settle_log "$thermal_settle_log" 60 \
    "$maximum_settle_sample_gap_seconds"
  settle_samples=$THERMAL_LOG_SAMPLES
  settle_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
  settle_telemetry_gaps=$THERMAL_LOG_GAPS
  cold_receipts_json=$(
    for receipt in "$out"/agents/agent-*.cold.json; do
      [[ -s "$receipt" ]] || {
        echo "missing DeepSeek cold receipt: $receipt" >&2
        exit 1
      }
      jq -n --arg name "$(basename "$receipt")" \
        --arg sha256 "$(sha256_file "$receipt")" \
        '{name:$name,sha256:$sha256}'
    done | jq -s 'sort_by(.name)'
  )
  jq -e '
    length == 4
    and ([.[].name] | unique | length) == 4
    and all(.[];
      (.name | test("^agent-[1-4]\\.cold\\.json$"))
      and (.sha256 | test("^[0-9a-f]{64}$")))
  ' <<<"$cold_receipts_json" >/dev/null
  jq -n --arg status pass --arg phase "deepseek-wave-$wave" \
    --arg settle_log_sha256 "$(sha256_file "$thermal_settle_log")" \
    --arg measurement_log_sha256 "$(sha256_file "$thermal_measurement_log")" \
    --argjson settle_seconds 60 \
    --argjson settle_duration_seconds "$settle_duration_seconds" \
    --argjson settle_samples "$settle_samples" \
    --argjson measurement_samples "$measurement_samples" \
    --argjson measurement_duration_seconds "$measurement_duration_seconds" \
    --argjson sample_interval_seconds "$sample_interval_seconds" \
    --argjson maximum_sample_gap_seconds "$maximum_sample_gap_seconds" \
    --argjson settle_sample_interval_seconds "$settle_sample_interval_seconds" \
    --argjson maximum_settle_sample_gap_seconds "$maximum_settle_sample_gap_seconds" \
    --argjson non_nominal_measurement_samples "$non_nominal_measurement_samples" \
    --argjson settle_telemetry_gaps "$settle_telemetry_gaps" \
    --argjson telemetry_gaps "$telemetry_gaps" \
    --argjson cold_receipts "$cold_receipts_json" \
    '{status:$status,phase:$phase,required_state:"nominal",runtime_preflight:"pass",
      measurement_scope:"cold-cohort",cold_receipts:$cold_receipts,
      settle_seconds:$settle_seconds,settle_duration_seconds:$settle_duration_seconds,
      settle_samples:$settle_samples,measurement_samples:$measurement_samples,
      measurement_duration_seconds:$measurement_duration_seconds,
      sample_interval_seconds:$sample_interval_seconds,
      maximum_sample_gap_seconds:$maximum_sample_gap_seconds,
      settle_sample_interval_seconds:$settle_sample_interval_seconds,
      maximum_settle_sample_gap_seconds:$maximum_settle_sample_gap_seconds,
      non_nominal_measurement_samples:$non_nominal_measurement_samples,
      settle_telemetry_gaps:$settle_telemetry_gaps,
      telemetry_gaps:$telemetry_gaps,settle_log_sha256:$settle_log_sha256,
      measurement_log_sha256:$measurement_log_sha256}' >"$thermal_summary"
  jq -e --slurpfile contract "$agentic_prompt_contract" \
    --arg prompt_contract_sha256 "$agentic_prompt_contract_sha" \
    --arg prompt_provenance_sha256 "$agentic_prompt_provenance_sha" \
    -f scripts/deepseek4_full_context_receipt.jq \
    "$out/summary.json.tmp" >/dev/null
  mv "$out/summary.json.tmp" "$out/summary.json"
  sha256_file "$out/summary.json" >"$out/summary.json.sha256"
  finish_server_phase
  cooperative_prefill_transactions=$(rg -c \
    'DeepSeek-V4 cooperative prefill complete' "$current_log" || true)
  [[ "$cooperative_prefill_transactions" =~ ^[0-9]+$ ]]
  ((cooperative_prefill_transactions > 0)) || {
    echo "DeepSeek wave $wave observed no cooperative warm-prefill transaction" >&2
    return 1
  }
  decode_cohort_transactions=$(rg -c \
    'DeepSeek-V4 exact decode cohort selected' "$current_log" || true)
  [[ "$decode_cohort_transactions" =~ ^[0-9]+$ ]]
  ((decode_cohort_transactions > 0)) || {
    echo "DeepSeek wave $wave observed no exact warm B=4 decode transaction" >&2
    return 1
  }
  cold_prefill_rates_json=$(
    for request_id in 1 2 3 4; do
      rate=$(rg "DeepSeek-V4 prefill complete request_id=${request_id}( |$)" "$current_log" \
        | head -1 | sed -n 's/.*tokens_per_second=\([^ ]*\).*/\1/p')
      [[ "$rate" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
        echo "missing DeepSeek cold prefill rate for request $request_id" >&2
        exit 1
      }
      printf '%s\n' "$rate"
    done | jq -Rsc 'split("\n") | map(select(length > 0) | tonumber)'
  )
  jq -e 'length == 4 and all(.[]; type == "number" and . > 0)' \
    <<<"$cold_prefill_rates_json" >/dev/null
  jq -n --argjson wave "$wave" --arg binary_sha256 "$binary_sha" \
    --arg model_sha256 "$DEEPSEEK_MODEL_SHA256" \
    --arg summary_sha256 "$(cat "$out/summary.json.sha256")" \
    --arg server_log_sha256 "$(cat "$current_dir/server.log.sha256")" \
    --argjson cooperative_prefill_transactions "$cooperative_prefill_transactions" \
    --argjson decode_cohort_transactions "$decode_cohort_transactions" \
    --argjson cold_prefill_tokens_per_second "$cold_prefill_rates_json" \
    --slurpfile thermal "$thermal_summary" \
    --slurpfile receipt "$out/summary.json" \
    '{wave:$wave,status:"pass",binary_sha256:$binary_sha256,model_sha256:$model_sha256,ready_http:200,fatal_log_signatures:0,summary_sha256:$summary_sha256,server_log_sha256:$server_log_sha256,cooperative_prefill_transactions:$cooperative_prefill_transactions,decode_cohort_transactions:$decode_cohort_transactions,cold_prefill_tokens_per_second:$cold_prefill_tokens_per_second,thermal:$thermal[0],receipt:$receipt[0]}' \
    >"$out/envelope.json.tmp"
  mv "$out/envelope.json.tmp" "$out/envelope.json"
  scripts/verify_macos_thermal_receipt.sh "$wave" "$out/envelope.json" \
    "$thermal_summary" "$thermal_measurement_log" "$thermal_settle_log" \
    "$out/agents"
}

run_deepseek_release_gates() {
  # Run calibrated performance waves on a thermally settled host before the
  # 94K/119K functional workloads heat-soak the shared M5 runner.
  run_deepseek_wave 1
  run_deepseek_wave 2

  start_server deepseek process-a scripts/serve_deepseek4_opencode.sh \
    "$DEEPSEEK_MODEL" 18080 4 524288 8589934592
  BASE_URL="$current_url" MODEL="Deepseek v4 Flash 0731 Source" \
  MODEL_PATH="$DEEPSEEK_MODEL" FIXTURE_JSON="$OUT_ROOT/fixtures/public-347.json" \
  SERVER_LOG="$current_log" SERVER_PID="$server_pid" \
  OUT_DIR="$OUT_ROOT/deepseek/interactive" BINARY_PATH="$HF2Q_BIN" \
  BINARY_SHA256="$binary_sha" MODEL_SHA256="$DEEPSEEK_MODEL_SHA256" \
  HF2Q_MODEL_VERIFICATION_RECEIPT="$OUT_ROOT/deepseek/model-verification.json" \
  MAX_SLOTS=4 NO_PROGRESS_SECONDS=30 \
    scripts/test_deepseek4_interactive_overlap.sh \
      >"$OUT_ROOT/deepseek/interactive.stdout" \
      2>"$OUT_ROOT/deepseek/interactive.stderr"
  verify_sha256_sidecar "$OUT_ROOT/deepseek/interactive/summary.json"

  BASE_URL="$current_url" MODEL="Deepseek v4 Flash 0731 Source" \
  SERVER_LOG="$current_log" SERVER_PID="$server_pid" \
  OUT_DIR="$OUT_ROOT/deepseek/cached-suffix" EXPECTED_PATH="$PWD/README.md" \
  OVERLAP_TOOL_RESULT_PATH="$PWD/README.md" \
  CANCEL_TOOL_RESULT_PATH="$PWD/docs/ADR-042-deepseek-v4-flash-rust-native.md" \
  CURL_MAX_TIME_SECONDS=180 PREFILL_CHUNKS_BEFORE_CANCEL=3 \
  CANCEL_SETTLE_SECONDS=15 CANCEL_STABILITY_SECONDS=10 \
    scripts/test_deepseek4_cached_suffix.sh \
      >"$OUT_ROOT/deepseek/cached-suffix.stdout" \
      2>"$OUT_ROOT/deepseek/cached-suffix.stderr"
  verify_sha256_sidecar "$OUT_ROOT/deepseek/cached-suffix/summary.json"
  run_lifecycle deepseek 3230
  finish_server_phase
}

run_qwen_release_gates() {
  start_server qwen process-a scripts/serve_qwen36_opencode.sh \
    "$QWEN_MODEL" 18081 4
  BASE_URL="$current_url" SERVER_PID="$server_pid" SERVER_LOG="$current_log" \
  BINARY_PATH="$HF2Q_BIN" BINARY_SHA256="$binary_sha" \
  MODEL_PATH="$QWEN_MODEL" MODEL_SHA256="$QWEN_MODEL_SHA256" \
  HF2Q_MODEL_VERIFICATION_RECEIPT="$OUT_ROOT/qwen/model-verification.json" \
  FIXTURE_JSON="$OUT_ROOT/fixtures/public-347.json" \
  SHORT_FIXTURE_JSON="$OUT_ROOT/fixtures/public-short.json" MAX_SLOTS=4 \
  EXPECTED_PATH=/opt/hf2q/Cargo.toml \
  TOOL_RESULT_PATH="$PWD/Cargo.toml" \
  TOOL_RESULT_SUCCESS_PREFIX=$'Result from the completed read_file call. The call succeeded; use this result to answer the user without calling read_file again. File follows:\n' \
  AGENTIC_SYSTEM_PROMPT='You are an agentic coding assistant. Use the provided tools directly whenever they are needed. Never describe, imitate, or wrap a tool call in Markdown or a code fence.' \
  OUT_DIR="$OUT_ROOT/qwen/cumulative" scripts/test_qwen36_cumulative_release.sh \
    >"$OUT_ROOT/qwen/cumulative.stdout" 2>"$OUT_ROOT/qwen/cumulative.stderr"
  verify_sha256_sidecar "$OUT_ROOT/qwen/cumulative/cumulative-release-summary.json"
  run_lifecycle qwen 2800
  finish_server_phase

  start_server qwen process-b scripts/serve_qwen36_opencode.sh \
    "$QWEN_MODEL" 18081 1
  BASE_URL="$current_url" SERVER_PID="$server_pid" SERVER_LOG="$current_log" \
  BINARY_PATH="$HF2Q_BIN" BINARY_SHA256="$binary_sha" \
  MODEL_PATH="$QWEN_MODEL" MODEL_SHA256="$QWEN_MODEL_SHA256" \
  HF2Q_MODEL_VERIFICATION_RECEIPT="$OUT_ROOT/qwen/model-verification.json" \
  FIXTURE_JSON="$OUT_ROOT/fixtures/public-347.json" \
  SHORT_FIXTURE_JSON="$OUT_ROOT/fixtures/public-short.json" MAX_SLOTS=1 \
  REQUIRE_PROVENANCE=1 OUT_DIR="$OUT_ROOT/qwen/cancellation" \
    scripts/test_qwen36_prefill_cancellation.sh \
      >"$OUT_ROOT/qwen/cancellation.stdout" 2>"$OUT_ROOT/qwen/cancellation.stderr"
  verify_sha256_sidecar "$OUT_ROOT/qwen/cancellation/cancellation-summary.json"
  qwen36_validate_cancellation_transaction_counts \
    "$OUT_ROOT/qwen/cancellation/cancellation-summary.json"
  finish_server_phase
}

run_qwen38_long_decode_release_gate() {
  local out="$OUT_ROOT/qwen38/long-decode"
  local benchmark_dir="$out/benchmark"
  local thermal_dir="$out/thermal"
  local thermal_measurement_log="$thermal_dir/measurement.log"
  local thermal_settle_log="$thermal_dir/settle.log"
  local thermal_summary="$thermal_dir/summary.json"
  local harness_rc=0
  local thermal_rc=0
  local measurement_samples measurement_duration_seconds
  local non_nominal_measurement_samples fair_measurement_samples
  local over_limit_measurement_samples telemetry_gaps
  local settle_samples settle_duration_seconds settle_telemetry_gaps
  local sample_interval_seconds=2
  local maximum_sample_gap_seconds=5
  local settle_sample_interval_seconds=5
  local maximum_settle_sample_gap_seconds=8

  mkdir -p "$out" "$thermal_dir"
  ensure_guard_health
  require_no_model_runtime
  thermal_wait_for_nominal "$thermal_settle_log" \
    qwen38-long-decode-settle 60 900 "$settle_sample_interval_seconds"
  : >"$thermal_measurement_log"
  thermal_sample "$thermal_measurement_log" \
    qwen38-long-decode-measurement-start
  test "$THERMAL_STATE" = nominal

  SOURCE_SHA="$EXPECTED_SHA" CRATE_SHA256="$CRATE_SHA256" \
  BINARY_PATH="$HF2Q_BIN" BINARY_SHA256="$binary_sha" \
  MODEL_PATH="$QWEN38_MODEL" MODEL_SHA256="$QWEN38_MODEL_SHA256" \
  HF2Q_MODEL_VERIFICATION_RECEIPT="$OUT_ROOT/qwen38/model-verification.json" \
  OUT_DIR="$benchmark_dir" PORT=18083 \
    scripts/qwen38_long_decode_ab.sh \
      >"$out/benchmark.stdout" 2>"$out/benchmark.stderr" &
  wave_harness_pid=$!
  set +e
  thermal_monitor_fair_or_better_while_pid "$thermal_measurement_log" \
    qwen38-long-decode-measurement "$wave_harness_pid" \
    "$sample_interval_seconds"
  thermal_rc=$?
  if ((thermal_rc != 0)); then
    kill -TERM "$wave_harness_pid" 2>/dev/null || true
  fi
  wait "$wave_harness_pid"
  harness_rc=$?
  wave_harness_pid=''
  set -e
  ((thermal_rc == 0)) || {
    echo "Qwen3.8 long-decode thermal supervision failed" >&2
    return 1
  }
  ((harness_rc == 0)) || return "$harness_rc"
  require_no_model_runtime
  ensure_guard_health
  thermal_sample "$thermal_measurement_log" \
    qwen38-long-decode-measurement-end
  [[ "$THERMAL_STATE" == nominal || "$THERMAL_STATE" == fair ]]

  thermal_validate_fair_or_better_measurement_log "$thermal_measurement_log" \
    "$maximum_sample_gap_seconds"
  measurement_samples=$THERMAL_LOG_SAMPLES
  measurement_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
  non_nominal_measurement_samples=$THERMAL_LOG_NON_NOMINAL_SAMPLES
  fair_measurement_samples=$THERMAL_LOG_FAIR_SAMPLES
  over_limit_measurement_samples=$THERMAL_LOG_OVER_LIMIT_SAMPLES
  telemetry_gaps=$THERMAL_LOG_GAPS
  thermal_validate_settle_log "$thermal_settle_log" 60 \
    "$maximum_settle_sample_gap_seconds"
  settle_samples=$THERMAL_LOG_SAMPLES
  settle_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
  settle_telemetry_gaps=$THERMAL_LOG_GAPS

  jq -n --arg benchmark_summary_sha256 \
    "$(sha256_file "$benchmark_dir/summary.json")" \
    --arg settle_log_sha256 "$(sha256_file "$thermal_settle_log")" \
    --arg measurement_log_sha256 "$(sha256_file "$thermal_measurement_log")" \
    --argjson settle_seconds 60 \
    --argjson settle_duration_seconds "$settle_duration_seconds" \
    --argjson settle_samples "$settle_samples" \
    --argjson measurement_samples "$measurement_samples" \
    --argjson measurement_duration_seconds "$measurement_duration_seconds" \
    --argjson sample_interval_seconds "$sample_interval_seconds" \
    --argjson maximum_sample_gap_seconds "$maximum_sample_gap_seconds" \
    --argjson settle_sample_interval_seconds "$settle_sample_interval_seconds" \
    --argjson maximum_settle_sample_gap_seconds "$maximum_settle_sample_gap_seconds" \
    --argjson non_nominal_measurement_samples "$non_nominal_measurement_samples" \
    --argjson fair_measurement_samples "$fair_measurement_samples" \
    --argjson over_limit_measurement_samples "$over_limit_measurement_samples" \
    --argjson settle_telemetry_gaps "$settle_telemetry_gaps" \
    --argjson telemetry_gaps "$telemetry_gaps" \
    '{status:"pass",phase:"qwen38-long-decode",required_start_state:"nominal",
      maximum_measurement_state:"fair",
      runtime_preflight:"pass",measurement_scope:"full-abba-benchmark",
      benchmark_summary_sha256:$benchmark_summary_sha256,
      settle_seconds:$settle_seconds,settle_duration_seconds:$settle_duration_seconds,
      settle_samples:$settle_samples,measurement_samples:$measurement_samples,
      measurement_duration_seconds:$measurement_duration_seconds,
      sample_interval_seconds:$sample_interval_seconds,
      maximum_sample_gap_seconds:$maximum_sample_gap_seconds,
      settle_sample_interval_seconds:$settle_sample_interval_seconds,
      maximum_settle_sample_gap_seconds:$maximum_settle_sample_gap_seconds,
      non_nominal_measurement_samples:$non_nominal_measurement_samples,
      fair_measurement_samples:$fair_measurement_samples,
      over_limit_measurement_samples:$over_limit_measurement_samples,
      settle_telemetry_gaps:$settle_telemetry_gaps,telemetry_gaps:$telemetry_gaps,
      settle_log_sha256:$settle_log_sha256,
      measurement_log_sha256:$measurement_log_sha256}' >"$thermal_summary.tmp"
  mv "$thermal_summary.tmp" "$thermal_summary"
  jq -n --slurpfile benchmark "$benchmark_dir/summary.json" \
    --slurpfile thermal "$thermal_summary" \
    '{schema_version:1,status:"pass",benchmark:$benchmark[0],thermal:$thermal[0]}' \
    >"$out/receipt.json.tmp"
  mv "$out/receipt.json.tmp" "$out/receipt.json"
  bash scripts/verify_qwen38_long_decode_receipt.sh release "$out" \
    "$EXPECTED_SHA" "$CRATE_SHA256" "$binary_sha" "$QWEN38_MODEL_SHA256"
}

run_gemma_wave() {
  local phase=$1
  local agents=$2
  local out="$OUT_ROOT/gemma/$phase"
  # Four slots are the release-validated operator default and keep the shared
  # agentic latency limits. Eight slots are an explicitly experimental
  # correctness/aggregate-transaction-cap probe. Give only that probe a
  # functional completion envelope sized from the exact M5 Max discriminator
  # (25.279 s cold TTFT, 23.932 s worst tool-result response); the measured
  # values still remain in every per-agent receipt.
  mkdir -p "$out"
  if [[ "$agents" == 8 ]]; then
    MAX_COLD_TTFT_MS=40000 MAX_COLD_RESPONSE_MS=60000 \
    MAX_TOOL_RESULT_RESPONSE_MS=30000 \
    BASE_URL="$current_url" FAMILY=gemma4 AGENTS="$agents" \
    WAVE_ID="$phase" REQUIRE_COLD_FIRST=1 \
    OUT_DIR="$out/agents" scripts/test_full_context_agent_slots.sh \
      >"$out/summary.json.tmp" 2>"$out/harness.stderr"
  else
    BASE_URL="$current_url" FAMILY=gemma4 AGENTS="$agents" \
    WAVE_ID="$phase" REQUIRE_COLD_FIRST=1 \
    OUT_DIR="$out/agents" scripts/test_full_context_agent_slots.sh \
      >"$out/summary.json.tmp" 2>"$out/harness.stderr"
  fi
  jq -e --argjson agents "$agents" \
    '.status == "pass" and .family == "gemma4" and .concurrent_agents == $agents and .require_cold_first == 1 and all(.agents[]; .cold_cached_tokens == 0)' \
    "$out/summary.json.tmp" >/dev/null
  mv "$out/summary.json.tmp" "$out/summary.json"
  shasum -a 256 "$out/summary.json" >"$out/summary.json.sha256"
  shasum -c "$out/summary.json.sha256" >/dev/null
}

run_gemma_thermally_guarded_wave() {
  local wave=$1
  local agents=$2
  local phase phase_name name_pattern
  case "$wave:$agents" in
  1:4 | 2:4)
    phase="wave$wave"
    phase_name="gemma-wave-$wave"
    name_pattern='^agent-[1-4]\.cold\.json$'
    ;;
  eight-slots:8)
    phase=eight-slots
    phase_name=gemma-eight-slots
    name_pattern='^agent-[1-8]\.cold\.json$'
    ;;
  *)
    echo "unsupported Gemma thermal wave: wave=$wave agents=$agents" >&2
    return 2
    ;;
  esac
  local out="$OUT_ROOT/gemma/$phase"
  local thermal_dir="$out/thermal"
  local thermal_measurement_log="$thermal_dir/measurement.log"
  local thermal_settle_log="$thermal_dir/settle.log"
  local thermal_summary="$thermal_dir/summary.json"
  local harness_rc=0
  local measurement_samples measurement_duration_seconds
  local non_nominal_measurement_samples telemetry_gaps
  local settle_samples settle_duration_seconds settle_telemetry_gaps
  local cold_receipts_json
  local sample_interval_seconds=2
  local maximum_sample_gap_seconds=5
  local settle_sample_interval_seconds=5
  local maximum_settle_sample_gap_seconds=8

  mkdir -p "$out" "$thermal_dir"
  thermal_prepare_cold_receipt_dir "$out/agents"
  thermal_wait_for_nominal "$thermal_settle_log" "$phase_name-settle" \
    60 900 "$settle_sample_interval_seconds"
  : >"$thermal_measurement_log"
  thermal_sample "$thermal_measurement_log" "$phase_name-measurement-start"
  test "$THERMAL_STATE" = nominal
  run_gemma_wave "$phase" "$agents" &
  wave_harness_pid=$!

  # Keep thermal supervision in the release-gate process. A separate monitor
  # used to require a stop-file/join handoff after the harness completed; a
  # passing exact-artifact wave could then fail between writing that sentinel
  # and publishing its terminal thermal receipt. Foreground sampling has one
  # owner and covers the same start-to-harness-exit interval without that
  # process-lifecycle race.
  if ! thermal_monitor_nominal_while_pid "$thermal_measurement_log" \
    "$phase_name-measurement" "$wave_harness_pid" \
    "$sample_interval_seconds"; then
    echo "Gemma wave $phase thermal supervision failed" >&2
    # Closing the server makes the harness's direct curl children return;
    # the harness then reaps its complete process tree itself.
    stop_server || true
    wait "$wave_harness_pid" 2>/dev/null || true
    wave_harness_pid=""
    return 1
  fi

  set +e
  wait "$wave_harness_pid"
  harness_rc=$?
  set -e
  wave_harness_pid=""
  ((harness_rc == 0)) || return "$harness_rc"
  # Append the terminal sample after the harness has been reaped. No other
  # process writes this log, so the end marker is necessarily the final row.
  thermal_sample "$thermal_measurement_log" "$phase_name-measurement-end"
  test "$THERMAL_STATE" = nominal

  thermal_validate_measurement_log "$thermal_measurement_log" \
    "$maximum_sample_gap_seconds"
  measurement_samples=$THERMAL_LOG_SAMPLES
  measurement_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
  non_nominal_measurement_samples=$THERMAL_LOG_NON_NOMINAL_SAMPLES
  telemetry_gaps=$THERMAL_LOG_GAPS
  thermal_validate_settle_log "$thermal_settle_log" 60 \
    "$maximum_settle_sample_gap_seconds"
  settle_samples=$THERMAL_LOG_SAMPLES
  settle_duration_seconds=$THERMAL_LOG_DURATION_SECONDS
  settle_telemetry_gaps=$THERMAL_LOG_GAPS
  cold_receipts_json=$(
    for receipt in "$out"/agents/agent-*.cold.json; do
      [[ -s "$receipt" ]] || {
        echo "missing Gemma cold receipt: $receipt" >&2
        exit 1
      }
      jq -n --arg name "$(basename "$receipt")" \
        --arg sha256 "$(sha256_file "$receipt")" \
        '{name:$name,sha256:$sha256}'
    done | jq -s 'sort_by(.name)'
  )
  jq -e --arg name_pattern "$name_pattern" --argjson agents "$agents" '
    length == $agents
    and ([.[].name] | unique | length) == $agents
    and all(.[];
      (.name | test($name_pattern))
      and (.sha256 | test("^[0-9a-f]{64}$")))
  ' <<<"$cold_receipts_json" >/dev/null
  jq -n --arg status pass --arg phase "$phase_name" \
    --arg settle_log_sha256 "$(sha256_file "$thermal_settle_log")" \
    --arg measurement_log_sha256 "$(sha256_file "$thermal_measurement_log")" \
    --argjson settle_seconds 60 \
    --argjson settle_duration_seconds "$settle_duration_seconds" \
    --argjson settle_samples "$settle_samples" \
    --argjson measurement_samples "$measurement_samples" \
    --argjson measurement_duration_seconds "$measurement_duration_seconds" \
    --argjson sample_interval_seconds "$sample_interval_seconds" \
    --argjson maximum_sample_gap_seconds "$maximum_sample_gap_seconds" \
    --argjson settle_sample_interval_seconds "$settle_sample_interval_seconds" \
    --argjson maximum_settle_sample_gap_seconds "$maximum_settle_sample_gap_seconds" \
    --argjson non_nominal_measurement_samples "$non_nominal_measurement_samples" \
    --argjson settle_telemetry_gaps "$settle_telemetry_gaps" \
    --argjson telemetry_gaps "$telemetry_gaps" \
    --argjson concurrent_agents "$agents" \
    --argjson cold_receipts "$cold_receipts_json" \
    '{status:$status,phase:$phase,concurrent_agents:$concurrent_agents,
      required_state:"nominal",
      measurement_scope:"full-agent-wave",cold_receipts:$cold_receipts,
      settle_seconds:$settle_seconds,settle_duration_seconds:$settle_duration_seconds,
      settle_samples:$settle_samples,measurement_samples:$measurement_samples,
      measurement_duration_seconds:$measurement_duration_seconds,
      sample_interval_seconds:$sample_interval_seconds,
      maximum_sample_gap_seconds:$maximum_sample_gap_seconds,
      settle_sample_interval_seconds:$settle_sample_interval_seconds,
      maximum_settle_sample_gap_seconds:$maximum_settle_sample_gap_seconds,
      non_nominal_measurement_samples:$non_nominal_measurement_samples,
      settle_telemetry_gaps:$settle_telemetry_gaps,telemetry_gaps:$telemetry_gaps,
      settle_log_sha256:$settle_log_sha256,
      measurement_log_sha256:$measurement_log_sha256}' >"$thermal_summary"
  jq --slurpfile thermal "$thermal_summary" '. + {thermal:$thermal[0]}' \
    "$out/summary.json" >"$out/summary.json.tmp"
  mv "$out/summary.json.tmp" "$out/summary.json"
  shasum -a 256 "$out/summary.json" >"$out/summary.json.sha256"
  shasum -c "$out/summary.json.sha256" >/dev/null
  bash scripts/verify_gemma4_wave_thermal_receipt.sh "$wave" \
    "$out/summary.json" "$thermal_summary" "$thermal_measurement_log" \
    "$thermal_settle_log" "$out/agents"
  if [[ "$wave" == eight-slots ]]; then
    jq -e -f scripts/gemma4_eight_slot_receipt.jq \
      "$out/summary.json" >/dev/null
  fi
}

capture_gemma_heap() {
  local phase=$1
  ensure_guard_health
  qwen36_capture_heap_summary "$server_pid" \
    "$OUT_ROOT/gemma/heap-$phase.txt" "$OUT_ROOT/gemma/heap-$phase.json"
}

write_gemma_transaction_receipt() {
  local phase=$1
  local log=$2
  local out="$OUT_ROOT/gemma/transactions-$phase.json"
  local rows="$OUT_ROOT/gemma/transactions-$phase.rows"
  local max_rows multi_slot nonaligned
  rg 'Gemma4 (bounded prefill transaction complete|installed prefills advanced in one aggregate-bounded transaction|stable agent suffixes prefilled in one multi-slot body)' \
    "$log" >"$rows"
  [[ -s "$rows" ]] || return 1
  max_rows=$(sed -n -E 's/.*(advanced_tokens|suffix_tokens)=([0-9]+).*/\2/p' "$rows" |
    sort -n | tail -1)
  [[ "$max_rows" =~ ^[0-9]+$ ]] && ((max_rows <= 4096)) || return 1
  multi_slot=$(rg -c 'requests=([2-9]|[1-9][0-9]+)( |$)' "$rows" || true)
  ((multi_slot > 0)) || return 1
  nonaligned=$(sed -n -E 's/.*(advanced_tokens|suffix_tokens)=([0-9]+).*/\2/p' "$rows" |
    awk '$1 > 0 && $1 < 4096 {count++} END {print count + 0}')
  ((nonaligned > 0)) || return 1
  jq -n --arg status pass --arg phase "$phase" --arg rows_sha256 "$(sha256_file "$rows")" \
    --argjson max_transaction_rows "$max_rows" --argjson multi_slot_transactions "$multi_slot" \
    --argjson nonaligned_transactions "$nonaligned" \
    '{status:$status,phase:$phase,transaction_cap_rows:4096,max_transaction_rows:$max_transaction_rows,multi_slot_transactions:$multi_slot_transactions,nonaligned_transactions:$nonaligned_transactions,rows_sha256:$rows_sha256}' \
    >"$out.tmp"
  jq -e '.status == "pass" and .max_transaction_rows <= .transaction_cap_rows and .multi_slot_transactions > 0 and .nonaligned_transactions > 0' \
    "$out.tmp" >/dev/null
  mv "$out.tmp" "$out"
}

run_gemma_release_gates() {
  # The four-slot latency waves are calibration work, not soak work. Run them
  # first on one fresh server, after a nominal settle for each wave, and keep
  # thermal monitoring live through every cold/cached/tool-result turn. The
  # destructive 175K overlap and 120K lifecycle follow only after both limits
  # have passed.
  require_no_model_runtime
  start_server gemma process-a scripts/serve_gemma4_opencode.sh \
    "$GEMMA_MODEL" 18082 4
  capture_gemma_heap baseline
  run_gemma_thermally_guarded_wave 1 4
  capture_gemma_heap post-wave1
  run_gemma_thermally_guarded_wave 2 4
  capture_gemma_heap post-wave2
  BASE_URL="$current_url" SERVER_PID="$server_pid" SERVER_LOG="$current_log" \
  BINARY_PATH="$HF2Q_BIN" BINARY_SHA256="$binary_sha" \
  MODEL_PATH="$GEMMA_MODEL" MODEL_SHA256="$GEMMA_MODEL_SHA256" MAX_SLOTS=4 \
  HF2Q_MODEL_VERIFICATION_RECEIPT="$OUT_ROOT/gemma/model-verification.json" \
  CURL_MAX_TIME_SECONDS=1800 CANCELLATION_WAIT_SECONDS=180 \
  OUT_DIR="$OUT_ROOT/gemma/overlap" scripts/test_gemma4_long_short_overlap.sh \
    >"$OUT_ROOT/gemma/overlap.stdout" 2>"$OUT_ROOT/gemma/overlap.stderr"
  verify_sha256_sidecar "$OUT_ROOT/gemma/overlap/summary.json"
  capture_gemma_heap post-overlap
  run_lifecycle gemma 2800
  capture_gemma_heap post-lifecycle
  jq -e -s '
    length == 5
    and all(.[]; .command_buffer_objects == 0 and .command_buffer_impls == 0)
    and all(.[]; .cfstring_count >= 0 and .autoreleasepool_content_count >= 0)
    and (.[1].cfstring_count - .[0].cfstring_count <= 256)
    and (.[2].cfstring_count - .[1].cfstring_count <= 256)
    and (.[2].cfstring_count - .[0].cfstring_count <= 512)
    and (.[3].cfstring_count - .[2].cfstring_count <= 512)
    and (.[4].cfstring_count - .[3].cfstring_count <= 512)
    and (.[4].cfstring_count - .[2].cfstring_count <= 1024)
    and (.[1].autoreleasepool_content_count - .[0].autoreleasepool_content_count <= 8)
    and (.[2].autoreleasepool_content_count - .[1].autoreleasepool_content_count <= 8)
    and (.[2].autoreleasepool_content_count - .[0].autoreleasepool_content_count <= 16)
    and (.[3].autoreleasepool_content_count - .[2].autoreleasepool_content_count <= 8)
    and (.[4].autoreleasepool_content_count - .[3].autoreleasepool_content_count <= 8)
    and (.[4].autoreleasepool_content_count - .[2].autoreleasepool_content_count <= 16)
  ' "$OUT_ROOT/gemma/heap-baseline.json" \
    "$OUT_ROOT/gemma/heap-post-wave1.json" \
    "$OUT_ROOT/gemma/heap-post-wave2.json" \
    "$OUT_ROOT/gemma/heap-post-overlap.json" \
    "$OUT_ROOT/gemma/heap-post-lifecycle.json" >/dev/null
  jq -n \
    --slurpfile baseline "$OUT_ROOT/gemma/heap-baseline.json" \
    --slurpfile wave1 "$OUT_ROOT/gemma/heap-post-wave1.json" \
    --slurpfile wave2 "$OUT_ROOT/gemma/heap-post-wave2.json" \
    --slurpfile overlap "$OUT_ROOT/gemma/heap-post-overlap.json" \
    --slurpfile lifecycle "$OUT_ROOT/gemma/heap-post-lifecycle.json" \
    '{status:"pass",snapshot_order:["baseline","post_wave1","post_wave2","post_overlap","post_lifecycle"],snapshots:{baseline:$baseline[0],post_wave1:$wave1[0],post_wave2:$wave2[0],post_overlap:$overlap[0],post_lifecycle:$lifecycle[0]}}' \
    >"$OUT_ROOT/gemma/heap-summary.json"
  jq -e '.status == "pass" and all(.snapshots[]; .command_buffer_objects == 0 and .command_buffer_impls == 0)' \
    "$OUT_ROOT/gemma/heap-summary.json" >/dev/null
  finish_server_phase
  write_gemma_transaction_receipt four-slots "$current_log"

  start_server gemma process-b scripts/serve_gemma4_opencode.sh \
    "$GEMMA_MODEL" 18082 8
  # The experimental eight-slot correctness wave runs after the destructive
  # four-slot soak, so it gets its own independent Nominal settle and
  # continuous full-wave receipt rather than inheriting the prior host state.
  run_gemma_thermally_guarded_wave eight-slots 8
  finish_server_phase
  write_gemma_transaction_receipt eight-slots "$current_log"

  ensure_guard_health
  mkdir -p "$OUT_ROOT/gemma/parity"
  HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF="$GEMMA_MODEL" \
    cargo test --release --locked --bin hf2q slot_aware_n4_per_slot_parity_vs_serial -- \
      --test-threads=1 >"$OUT_ROOT/gemma/parity/n4.log" 2>&1
  HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF="$GEMMA_MODEL" \
    HF2Q_CROSS_SLOT_ADMIT=1 HF2Q_ADMIT_COALESCE_US=25000 \
    HF2Q_GEMMA_N8_PARITY_MAX_TOKENS=24 \
    HF2Q_GEMMA_N8_PARITY_ROUNDS=25 \
    cargo test --release --locked --bin hf2q slot_aware_n8_per_slot_parity_vs_serial -- \
      --test-threads=1 --nocapture >"$OUT_ROOT/gemma/parity/n8.log" 2>&1
  HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF="$GEMMA_MODEL" \
    HF2Q_CROSS_SLOT_ADMIT=1 HF2Q_ADMIT_COALESCE_US=25000 \
    HF2Q_GEMMA_N8_PARITY_MAX_TOKENS=1 \
    HF2Q_GEMMA_N8_PARITY_ROUNDS=25 \
    cargo test --release --locked --bin hf2q slot_aware_n8_per_slot_parity_vs_serial -- \
      --test-threads=1 --nocapture >"$OUT_ROOT/gemma/parity/n8-seed-budget.log" 2>&1
  HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF="$GEMMA_MODEL" \
    HF2Q_HYBRID_KV=1 HF2Q_USE_DENSE=0 HF2Q_TQ_CODEBOOK_BITS=8 \
    HF2Q_GEMMA_N8_EXPECTED_KV_REGIME=hybrid \
    HF2Q_GEMMA_N8_PREFILL_REPEATS=64 HF2Q_GEMMA_N8_RESUME_REPEATS=16 \
    cargo test --release --locked --bin hf2q \
      gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant -- \
      --test-threads=1 --nocapture >"$OUT_ROOT/gemma/parity/n8-tiny-hybrid.log" 2>&1
  HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF="$GEMMA_MODEL" \
    HF2Q_HYBRID_KV=0 HF2Q_USE_DENSE=0 HF2Q_TQ_CODEBOOK_BITS=8 \
    HF2Q_GEMMA_N8_EXPECTED_KV_REGIME=full-tq \
    HF2Q_GEMMA_N8_PREFILL_REPEATS=64 \
    HF2Q_GEMMA_N8_RESUME_REPEATS=16 \
    cargo test --release --locked --bin hf2q \
      gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant -- \
      --test-threads=1 --nocapture >"$OUT_ROOT/gemma/parity/n8-tiny-full-tq.log" 2>&1
  HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF="$GEMMA_MODEL" \
    cargo test --release --locked --bin hf2q \
      gemma_fresh_and_reused_4096_8193_bounded_outputs_match -- \
      --test-threads=1 >"$OUT_ROOT/gemma/parity/boundary-tail.log" 2>&1
  HF2Q_KV_PERSIST_PHASE_D=1 HF2Q_KV_PERSIST_E2E_MODEL_PATH="$GEMMA_MODEL" \
    cargo test --release --locked --test lcp_partial_prefill_byte_identity \
      gemma_hybrid_long_resume_byte_identity -- --test-threads=1 \
      >"$OUT_ROOT/gemma/parity/long-resume.log" 2>&1
  ensure_guard_health
  jq -n --arg status pass \
    --arg n4_sha256 "$(sha256_file "$OUT_ROOT/gemma/parity/n4.log")" \
    --arg n8_sha256 "$(sha256_file "$OUT_ROOT/gemma/parity/n8.log")" \
    --arg n8_seed_budget_sha256 "$(sha256_file "$OUT_ROOT/gemma/parity/n8-seed-budget.log")" \
    --arg n8_tiny_hybrid_sha256 "$(sha256_file "$OUT_ROOT/gemma/parity/n8-tiny-hybrid.log")" \
    --arg n8_tiny_full_tq_sha256 "$(sha256_file "$OUT_ROOT/gemma/parity/n8-tiny-full-tq.log")" \
    --arg boundary_tail_sha256 "$(sha256_file "$OUT_ROOT/gemma/parity/boundary-tail.log")" \
    --arg long_resume_sha256 "$(sha256_file "$OUT_ROOT/gemma/parity/long-resume.log")" \
    '{status:$status,profile:"release",n4_exact_output_parity:true,n8_exact_output_parity:true,n8_cross_slot_admit:true,n8_max_tokens:24,n8_rounds:25,n8_seed_budget_exact_output_parity:true,n8_seed_budget_max_tokens:1,n8_seed_budget_rounds:25,n8_tiny_hybrid_exact_output_parity:true,n8_tiny_full_tq_exact_output_parity:true,n8_tiny_prefill_rounds:64,n8_tiny_resume_rounds:16,fresh_and_reused_4096_8193_bounded_output_parity:true,long_resume_exact_output_parity:true,n4_log_sha256:$n4_sha256,n8_log_sha256:$n8_sha256,n8_seed_budget_log_sha256:$n8_seed_budget_sha256,n8_tiny_hybrid_log_sha256:$n8_tiny_hybrid_sha256,n8_tiny_full_tq_log_sha256:$n8_tiny_full_tq_sha256,boundary_tail_log_sha256:$boundary_tail_sha256,long_resume_log_sha256:$long_resume_sha256}' \
    >"$OUT_ROOT/gemma/parity/summary.json"
  bash scripts/verify_gemma4_parity_receipt.sh \
    "$OUT_ROOT/gemma/parity/summary.json" "$OUT_ROOT/gemma/parity"
}

# The 100 GiB DeepSeek artifact runs first. Every process fully exits before
# another model is loaded, so this remains safe on a 128 GiB host.
run_deepseek_release_gates
run_gemma_release_gates
run_qwen_release_gates
run_qwen38_long_decode_release_gate

hf2q_release_verify_model "$DEEPSEEK_MODEL" "$DEEPSEEK_MODEL_SHA256" \
  "$OUT_ROOT/deepseek/model-verification.json"
hf2q_release_verify_model "$GEMMA_MODEL" "$GEMMA_MODEL_SHA256" \
  "$OUT_ROOT/gemma/model-verification.json"
hf2q_release_verify_model "$QWEN_MODEL" "$QWEN_MODEL_SHA256" \
  "$OUT_ROOT/qwen/model-verification.json"
hf2q_release_verify_model "$QWEN38_MODEL" "$QWEN38_MODEL_SHA256" \
  "$OUT_ROOT/qwen38/model-verification.json"
ensure_guard_health
pmset -g assertions > "$OUT_ROOT/power-assertions.after.txt"
power_guarded_ac=true
power_snapshot_manifest="$OUT_ROOT/power-event-snapshots.sha256"
power_snapshot_prefixes=()
while IFS= read -r prefix; do
  power_snapshot_prefixes+=("$prefix")
done < <(qwen36_release_power_snapshot_prefixes)
qwen36_write_power_snapshot_manifest \
  "$OUT_ROOT" "$power_snapshot_manifest" "${power_snapshot_prefixes[@]}"
qwen36_verify_power_snapshot_manifest \
  "$OUT_ROOT" "$power_snapshot_manifest" "${power_snapshot_prefixes[@]}"
power_snapshot_manifest_sha=$(sha256_file "$power_snapshot_manifest")
deepseek_bytes=$(stat -f '%z' "$DEEPSEEK_MODEL")
gemma_bytes=$(stat -f '%z' "$GEMMA_MODEL")
qwen_bytes=$(stat -f '%z' "$QWEN_MODEL")
qwen38_bytes=$(stat -f '%z' "$QWEN38_MODEL")
qwen36_validate_cancellation_transaction_counts \
  "$OUT_ROOT/qwen/cancellation/cancellation-summary.json"
bash scripts/verify_gemma4_parity_receipt.sh \
  "$OUT_ROOT/gemma/parity/summary.json" "$OUT_ROOT/gemma/parity"
jq -n \
  --arg status pass \
  --arg source_sha "$EXPECTED_SHA" \
  --arg crate_sha256 "$CRATE_SHA256" \
  --arg binary_sha256 "$binary_sha" \
  --arg power_event_snapshots_sha256 "$power_snapshot_manifest_sha" \
  --arg deepseek_path "$DEEPSEEK_MODEL" \
  --arg gemma_path "$GEMMA_MODEL" \
  --arg qwen_path "$QWEN_MODEL" \
  --arg qwen38_path "$QWEN38_MODEL" \
  --arg deepseek_sha "$DEEPSEEK_MODEL_SHA256" \
  --arg gemma_sha "$GEMMA_MODEL_SHA256" \
  --arg qwen_sha "$QWEN_MODEL_SHA256" \
  --arg qwen38_sha "$QWEN38_MODEL_SHA256" \
  --arg deepseek_lifecycle_sha "$(sha256_file "$OUT_ROOT/deepseek/lifecycle/summary.json")" \
  --arg deepseek_interactive_sha "$(sha256_file "$OUT_ROOT/deepseek/interactive/summary.json")" \
  --arg deepseek_cached_sha "$(sha256_file "$OUT_ROOT/deepseek/cached-suffix/summary.json")" \
  --arg deepseek_cooperative_sha "$(sha256_file "$OUT_ROOT/deepseek/cooperative-prefill/summary.json")" \
  --arg deepseek_decode_cohort_sha "$(sha256_file "$OUT_ROOT/deepseek/decode-cohort/summary.json")" \
  --arg deepseek_wave1_sha "$(sha256_file "$OUT_ROOT/deepseek/full-context-1/envelope.json")" \
  --arg deepseek_wave2_sha "$(sha256_file "$OUT_ROOT/deepseek/full-context-2/envelope.json")" \
  --arg deepseek_wave1_thermal_sha "$(sha256_file "$OUT_ROOT/deepseek/full-context-1/thermal/summary.json")" \
  --arg deepseek_wave2_thermal_sha "$(sha256_file "$OUT_ROOT/deepseek/full-context-2/thermal/summary.json")" \
  --arg deepseek_prompt_contract_sha "$agentic_prompt_contract_sha" \
  --arg deepseek_prompt_provenance_sha "$agentic_prompt_provenance_sha" \
  --arg deepseek_context_fixture_sha "$agentic_fixture_sha" \
  --arg deepseek_tool_result_sha "$(jq -er '.tool_result.sha256' "$agentic_prompt_contract")" \
  --arg gemma_lifecycle_sha "$(sha256_file "$OUT_ROOT/gemma/lifecycle/summary.json")" \
  --arg gemma_overlap_sha "$(sha256_file "$OUT_ROOT/gemma/overlap/summary.json")" \
  --arg gemma_wave1_sha "$(sha256_file "$OUT_ROOT/gemma/wave1/summary.json")" \
  --arg gemma_wave2_sha "$(sha256_file "$OUT_ROOT/gemma/wave2/summary.json")" \
  --arg gemma_wave1_thermal_sha "$(sha256_file "$OUT_ROOT/gemma/wave1/thermal/summary.json")" \
  --arg gemma_wave2_thermal_sha "$(sha256_file "$OUT_ROOT/gemma/wave2/thermal/summary.json")" \
  --arg gemma_wave8_sha "$(sha256_file "$OUT_ROOT/gemma/eight-slots/summary.json")" \
  --arg gemma_wave8_thermal_sha "$(sha256_file "$OUT_ROOT/gemma/eight-slots/thermal/summary.json")" \
  --arg gemma_transactions4_sha "$(sha256_file "$OUT_ROOT/gemma/transactions-four-slots.json")" \
  --arg gemma_transactions8_sha "$(sha256_file "$OUT_ROOT/gemma/transactions-eight-slots.json")" \
  --arg gemma_parity_sha "$(sha256_file "$OUT_ROOT/gemma/parity/summary.json")" \
  --arg gemma_heap_sha "$(sha256_file "$OUT_ROOT/gemma/heap-summary.json")" \
  --arg qwen_lifecycle_sha "$(sha256_file "$OUT_ROOT/qwen/lifecycle/summary.json")" \
  --arg qwen_cumulative_sha "$(sha256_file "$OUT_ROOT/qwen/cumulative/cumulative-release-summary.json")" \
  --arg qwen_cancellation_sha "$(sha256_file "$OUT_ROOT/qwen/cancellation/cancellation-summary.json")" \
  --arg qwen38_long_decode_sha "$(sha256_file "$OUT_ROOT/qwen38/long-decode/receipt.json")" \
  --argjson power_guarded_ac "$power_guarded_ac" \
  --argjson deepseek_bytes "$deepseek_bytes" \
  --argjson gemma_bytes "$gemma_bytes" \
  --argjson qwen_bytes "$qwen_bytes" \
  --argjson qwen38_bytes "$qwen38_bytes" \
  --slurpfile deepseek_lifecycle "$OUT_ROOT/deepseek/lifecycle/summary.json" \
  --slurpfile deepseek_interactive "$OUT_ROOT/deepseek/interactive/summary.json" \
  --slurpfile deepseek_cached "$OUT_ROOT/deepseek/cached-suffix/summary.json" \
  --slurpfile deepseek_cooperative "$OUT_ROOT/deepseek/cooperative-prefill/summary.json" \
  --slurpfile deepseek_decode_cohort "$OUT_ROOT/deepseek/decode-cohort/summary.json" \
  --slurpfile deepseek_wave1 "$OUT_ROOT/deepseek/full-context-1/envelope.json" \
  --slurpfile deepseek_wave2 "$OUT_ROOT/deepseek/full-context-2/envelope.json" \
  --slurpfile deepseek_prompt_provenance "$agentic_prompt_provenance" \
  --slurpfile gemma_lifecycle "$OUT_ROOT/gemma/lifecycle/summary.json" \
  --slurpfile gemma_overlap "$OUT_ROOT/gemma/overlap/summary.json" \
  --slurpfile gemma_wave1 "$OUT_ROOT/gemma/wave1/summary.json" \
  --slurpfile gemma_wave2 "$OUT_ROOT/gemma/wave2/summary.json" \
  --slurpfile gemma_wave8 "$OUT_ROOT/gemma/eight-slots/summary.json" \
  --slurpfile gemma_transactions4 "$OUT_ROOT/gemma/transactions-four-slots.json" \
  --slurpfile gemma_transactions8 "$OUT_ROOT/gemma/transactions-eight-slots.json" \
  --slurpfile gemma_parity "$OUT_ROOT/gemma/parity/summary.json" \
  --slurpfile gemma_heap "$OUT_ROOT/gemma/heap-summary.json" \
  --slurpfile qwen_lifecycle "$OUT_ROOT/qwen/lifecycle/summary.json" \
  --slurpfile qwen_cumulative "$OUT_ROOT/qwen/cumulative/cumulative-release-summary.json" \
  --slurpfile qwen_cancellation "$OUT_ROOT/qwen/cancellation/cancellation-summary.json" \
  --slurpfile qwen38_long_decode "$OUT_ROOT/qwen38/long-decode/receipt.json" \
  'if ($qwen_cancellation | length) != 1 then
    error("Qwen cancellation receipt must contain exactly one JSON document")
  else {
    status: $status,
    source_sha: $source_sha,
    crate_sha256: $crate_sha256,
    binary_sha256: $binary_sha256,
    power_guarded_ac: $power_guarded_ac,
    power_event_snapshots_sha256: $power_event_snapshots_sha256,
    models: {
      deepseek: {path: $deepseek_path, bytes: $deepseek_bytes, sha256: $deepseek_sha},
      gemma: {path: $gemma_path, bytes: $gemma_bytes, sha256: $gemma_sha},
      qwen: {path: $qwen_path, bytes: $qwen_bytes, sha256: $qwen_sha},
      qwen38: {path: $qwen38_path, bytes: $qwen38_bytes, sha256: $qwen38_sha}
    },
    fixtures: {
      deepseek_agentic: {
        contract_sha256:$deepseek_prompt_contract_sha,
        context_sha256:$deepseek_context_fixture_sha,
        tool_result_sha256:$deepseek_tool_result_sha,
        prompt_provenance_sha256:$deepseek_prompt_provenance_sha
      }
    },
    receipt_sha256: {
      deepseek:{lifecycle:$deepseek_lifecycle_sha,interactive:$deepseek_interactive_sha,cached_suffix:$deepseek_cached_sha,cooperative_prefill:$deepseek_cooperative_sha,decode_cohort:$deepseek_decode_cohort_sha,wave1:$deepseek_wave1_sha,wave2:$deepseek_wave2_sha,wave1_thermal:$deepseek_wave1_thermal_sha,wave2_thermal:$deepseek_wave2_thermal_sha,prompt_provenance:$deepseek_prompt_provenance_sha},
      gemma:{lifecycle:$gemma_lifecycle_sha,overlap:$gemma_overlap_sha,wave1:$gemma_wave1_sha,wave2:$gemma_wave2_sha,wave1_thermal:$gemma_wave1_thermal_sha,wave2_thermal:$gemma_wave2_thermal_sha,eight_slots:$gemma_wave8_sha,eight_slots_thermal:$gemma_wave8_thermal_sha,transactions4:$gemma_transactions4_sha,transactions8:$gemma_transactions8_sha,parity:$gemma_parity_sha,heap:$gemma_heap_sha},
      qwen:{lifecycle:$qwen_lifecycle_sha,cumulative:$qwen_cumulative_sha,cancellation:$qwen_cancellation_sha},
      qwen38:{long_decode:$qwen38_long_decode_sha}
    },
    families: {
      deepseek: {status:"pass",prompt_provenance:$deepseek_prompt_provenance[0],cooperative_prefill:$deepseek_cooperative[0],decode_cohort:$deepseek_decode_cohort[0],lifecycle:$deepseek_lifecycle[0],interactive_overlap:$deepseek_interactive[0],cached_suffix:$deepseek_cached[0],full_context_waves:[$deepseek_wave1[0],$deepseek_wave2[0]]},
      gemma: {status:"pass",lifecycle:$gemma_lifecycle[0],overlap_and_cancellation:$gemma_overlap[0],agent_waves:[$gemma_wave1[0],$gemma_wave2[0],$gemma_wave8[0]],transactions:[$gemma_transactions4[0],$gemma_transactions8[0]],parity:$gemma_parity[0],heap:$gemma_heap[0]},
      qwen: {status:"pass",lifecycle:$qwen_lifecycle[0],cumulative:$qwen_cumulative[0],cancellation:$qwen_cancellation[0]},
      qwen38: {status:"pass",long_decode:$qwen38_long_decode[0]}
    }
  } end' > "$OUT_ROOT/manifest.json.tmp"
mv "$OUT_ROOT/manifest.json.tmp" "$OUT_ROOT/manifest.json"
shasum -a 256 "$OUT_ROOT/manifest.json" >"$OUT_ROOT/manifest.json.sha256"
jq -e '
  (.power_event_snapshots_sha256 | test("^[0-9a-f]{64}$"))
  and all(.receipt_sha256[][]; test("^[0-9a-f]{64}$"))
' "$OUT_ROOT/manifest.json" >/dev/null
jq . "$OUT_ROOT/manifest.json"
