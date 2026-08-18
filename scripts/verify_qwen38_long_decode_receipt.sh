#!/usr/bin/env bash
set -euo pipefail

verification_scope=${1:?verification scope (benchmark or release) is required}
receipt_root=${2:?receipt directory is required}
expected_source_sha=${3:?expected source SHA is required}
expected_crate_sha256=${4:?expected crate SHA-256 is required}
expected_binary_sha256=${5:?expected binary SHA-256 is required}
expected_model_sha256=${6:?expected model SHA-256 is required}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"

for command in awk basename cmp find grep head jq mktemp rm sed shasum sort stat \
  tail tr wc; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done

case "$verification_scope" in
  benchmark)
    benchmark_dir=$receipt_root
    ;;
  release)
    benchmark_dir="$receipt_root/benchmark"
    ;;
  *)
    echo "verification scope must be benchmark or release" >&2
    exit 2
    ;;
esac

for digest in "$expected_crate_sha256" "$expected_binary_sha256" "$expected_model_sha256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "expected artifact identities must be lowercase SHA-256 digests" >&2
    exit 2
  }
done
[[ "$expected_source_sha" =~ ^[0-9a-f]{40}$ ]] || {
  echo "expected source identity must be a full lowercase Git SHA" >&2
  exit 2
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
file_bytes() {
  stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1" 2>/dev/null
}
float_close() {
  local observed=$1
  local expected=$2
  local tolerance=$3
  awk -v observed="$observed" -v expected="$expected" -v tolerance="$tolerance" '
    BEGIN {
      delta = observed - expected
      if (delta < 0) delta = -delta
      exit !(delta <= tolerance)
    }
  '
}
artifact_digest() {
  local trial_json=$1
  local name=$2
  jq -er --arg name "$name" '
    [.artifacts[] | select(.name == $name)]
    | if length == 1 then .[0].sha256 else error("artifact inventory mismatch") end
  ' "$trial_json"
}

summary="$benchmark_dir/summary.json"
[[ -s "$summary" && -s "$summary.sha256" ]] || {
  echo "Qwen3.8 benchmark summary or checksum is missing" >&2
  exit 1
}
summary_digest=$(sha256_file "$summary")
grep -qx "$summary_digest  summary.json" "$summary.sha256"
(
  cd "$benchmark_dir"
  shasum -a 256 -c summary.json.sha256 >/dev/null
)

jq -e \
  --arg source_sha "$expected_source_sha" \
  --arg crate_sha256 "$expected_crate_sha256" \
  --arg binary_sha256 "$expected_binary_sha256" \
  --arg model_sha256 "$expected_model_sha256" '
    .schema_version == 1
    and .status == "pass"
    and .benchmark == "qwen38-long-decode-gqa-q2"
    and .identity.source_sha == $source_sha
    and .identity.crate_sha256 == $crate_sha256
    and .identity.binary.sha256 == $binary_sha256
    and (.identity.binary.path | type) == "string" and (.identity.binary.path | length) > 0
    and (.identity.binary.file_identity | test("^[0-9]+:[0-9]+$"))
    and .identity.model.id == "Qwen3.8 27B"
    and .identity.model.sha256 == $model_sha256
    and (.identity.model.path | type) == "string" and (.identity.model.path | length) > 0
    and (.identity.model.file_identity | test("^[0-9]+:[0-9]+$"))
    and (.identity.model.bytes | type) == "number" and .identity.model.bytes > 0
    and .identity.prompt.path == "prompt.txt"
    and (.identity.prompt.sha256 | test("^[0-9a-f]{64}$"))
    and (.identity.prompt.bytes | type) == "number" and .identity.prompt.bytes > 0
    and .identity.prompt.padding_tokens == 105000
    and .identity.phase_log.path == "phase.log"
    and (.identity.phase_log.sha256 | test("^[0-9a-f]{64}$"))
    and (.identity.phase_log.bytes | type) == "number" and .identity.phase_log.bytes > 0
    and .identity.request.path == "request.json"
    and (.identity.request.sha256 | test("^[0-9a-f]{64}$"))
    and (.identity.hardware.model | type) == "string" and (.identity.hardware.model | length) > 0
    and (.identity.hardware.chip | type) == "string" and (.identity.hardware.chip | length) > 0
    and .identity.hardware.arch == "arm64"
    and (.identity.hardware.memory_bytes | type) == "number"
    and .identity.hardware.memory_bytes > 0
    and (.identity.hardware.os_version | type) == "string"
    and (.identity.hardware.os_version | length) > 0
    and .settings.temperature == 0
    and (.settings | has("seed") | not)
    and .settings.max_tokens == 512
    and .settings.stream == false
    and .settings.thinking == false
    and .settings.repetition_penalty == 1.0
    and .settings.min_prompt_tokens == 100000
    and .settings.max_prompt_tokens == 120000
    and .settings.trial_settle_seconds == 30
    and .trial_order == ["off","auto","auto","off"]
    and (.trials | type) == "array" and (.trials | length) == 4
    and .aggregate.minimum_improvement_percent == 15
    and (.aggregate.exact_output_sha256 | test("^[0-9a-f]{64}$"))
  ' "$summary" >/dev/null

for root_artifact in prompt.txt request.json phase.log; do
  [[ -s "$benchmark_dir/$root_artifact" ]] || {
    echo "Qwen3.8 benchmark root artifact is missing: $root_artifact" >&2
    exit 1
  }
done
test "$(sha256_file "$benchmark_dir/prompt.txt")" = \
  "$(jq -er .identity.prompt.sha256 "$summary")"
test "$(file_bytes "$benchmark_dir/prompt.txt")" = \
  "$(jq -er .identity.prompt.bytes "$summary")"
test "$(sha256_file "$benchmark_dir/request.json")" = \
  "$(jq -er .identity.request.sha256 "$summary")"
test "$(sha256_file "$benchmark_dir/phase.log")" = \
  "$(jq -er .identity.phase_log.sha256 "$summary")"
test "$(file_bytes "$benchmark_dir/phase.log")" = \
  "$(jq -er .identity.phase_log.bytes "$summary")"
jq -e '
  ([.. | objects | has("seed")] | any | not)
  and .model == "Qwen3.8 27B"
  and .temperature == 0
  and .max_tokens == 512
  and .stream == false
  and .hf2q_enable_thinking == false
  and .repetition_penalty == 1.0
  and (.messages | length) == 2
  and .messages[0].role == "system"
  and .messages[1].role == "user"
  and (.messages[1].content | type) == "string"
' "$benchmark_dir/request.json" >/dev/null
tmp=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen38-receipt.XXXXXX")
cleanup_verifier() { rm -rf "$tmp"; }
trap cleanup_verifier EXIT
jq -jr '.messages[1].content' "$benchmark_dir/request.json" >"$tmp/request-prompt.txt"
cmp "$benchmark_dir/prompt.txt" "$tmp/request-prompt.txt"
expected_phase="$tmp/expected-phase"
actual_phase="$tmp/actual-phase"
phase_trial=0
for phase_mode in off auto auto off; do
  phase_trial=$((phase_trial + 1))
  for phase_event in trial-start request-start request-end trial-end; do
    printf '%s\t%s\t%s\n' "$phase_trial" "$phase_mode" "$phase_event"
  done
done >"$expected_phase"
awk -F '\t' '
  NF != 4 || $1 !~ /^[0-9]+$/ { exit 1 }
  NR > 1 && $1 < previous { exit 1 }
  { previous = $1; print $2 "\t" $3 "\t" $4 }
  END { if (NR != 16) exit 1 }
' "$benchmark_dir/phase.log" >"$actual_phase"
cmp "$expected_phase" "$actual_phase"

off_tps=()
auto_tps=()
semantic_sha=''
request_sha=$(jq -er .identity.request.sha256 "$summary")
binary_file_identity=$(jq -er .identity.binary.file_identity "$summary")
model_file_identity=$(jq -er .identity.model.file_identity "$summary")
trial_index=0
for mode in off auto auto off; do
  trial_index=$((trial_index + 1))
  trial_dir="$benchmark_dir/trial-${trial_index}-${mode}"
  trial_json="$trial_dir/trial.json"
  [[ -s "$trial_json" ]] || {
    echo "Qwen3.8 trial receipt is missing: trial-${trial_index}-${mode}" >&2
    exit 1
  }
  expected_inventory="$tmp/expected-inventory-$trial_index"
  actual_inventory="$tmp/actual-inventory-$trial_index"
  printf '%s\n' curl.metrics environment.txt models.json readyz.json request.json \
    response.json semantic.json server.log settle.log trial.json | sort >"$expected_inventory"
  find "$trial_dir" -mindepth 1 -maxdepth 1 -type f -exec basename {} \; \
    | sort >"$actual_inventory"
  cmp "$expected_inventory" "$actual_inventory"

  jq -S ".trials[$((trial_index - 1))]" "$summary" >"$tmp/embedded-trial.json"
  jq -S . "$trial_json" >"$tmp/raw-trial.json"
  cmp "$tmp/embedded-trial.json" "$tmp/raw-trial.json"
  jq -e --argjson index "$trial_index" --arg mode "$mode" \
    --arg binary_sha256 "$expected_binary_sha256" \
    --arg binary_file_identity "$binary_file_identity" \
    --arg model_sha256 "$expected_model_sha256" \
    --arg model_file_identity "$model_file_identity" \
    --arg request_sha256 "$request_sha" '
      .index == $index and .mode == $mode and .status == "pass"
      and .binary_sha256 == $binary_sha256
      and .binary_file_identity == $binary_file_identity
      and .model_sha256 == $model_sha256
      and .model_file_identity == $model_file_identity
      and .request_sha256 == $request_sha256
      and (.semantic_sha256 | test("^[0-9a-f]{64}$"))
      and (.prompt_tokens | type) == "number"
      and .prompt_tokens >= 100000 and .prompt_tokens <= 120000
      and .completion_tokens == 512
      and .finish_reason == "length"
      and (.decode_seconds | type) == "number" and .decode_seconds > 0
      and (.decode_tokens_per_second | type) == "number"
      and .decode_tokens_per_second > 0
      and (.artifacts | type) == "array" and (.artifacts | length) == 9
      and ([.artifacts[].name] | unique | length) == 9
      and all(.artifacts[]; (.sha256 | test("^[0-9a-f]{64}$")))
    ' "$trial_json" >/dev/null

  for artifact in request.json response.json semantic.json curl.metrics server.log readyz.json models.json environment.txt settle.log; do
    [[ -s "$trial_dir/$artifact" ]] || {
      echo "Qwen3.8 raw trial artifact is missing or empty: $artifact" >&2
      exit 1
    }
    test "$(sha256_file "$trial_dir/$artifact")" = \
      "$(artifact_digest "$trial_json" "$artifact")"
  done
  cmp "$benchmark_dir/request.json" "$trial_dir/request.json"
  jq -e '
    ([.. | objects | has("seed")] | any | not)
    and .temperature == 0 and .max_tokens == 512 and .stream == false
    and .hf2q_enable_thinking == false and .repetition_penalty == 1.0
  ' "$trial_dir/request.json" >/dev/null
  [[ "$(grep -c '^HF2Q_QWEN_GQA_Q2=' "$trial_dir/environment.txt")" == 1 ]]
  grep -qx "HF2Q_QWEN_GQA_Q2=$mode" "$trial_dir/environment.txt"
  grep -qx 'HF2Q_PIPELINE_PREWARM_LOG=1' "$trial_dir/environment.txt"
  grep -qx 'QWEN38_VISION=off' "$trial_dir/environment.txt"
  [[ "$(wc -l <"$trial_dir/environment.txt" | tr -d ' ')" == 3 ]]
  grep -qx 'http_code=200' "$trial_dir/curl.metrics"
  total_seconds=$(sed -n 's/^total_seconds=//p' "$trial_dir/curl.metrics")
  [[ "$total_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]
  awk -v seconds="$total_seconds" 'BEGIN { exit !(seconds > 0) }'
  jq -e '.ready == true' "$trial_dir/readyz.json" >/dev/null
  jq -e '
    (.object == "list")
    and ([.data[] | select(.loaded == true) | .id] == ["Qwen3.8 27B"])
  ' "$trial_dir/models.json" >/dev/null
  thermal_validate_settle_log "$trial_dir/settle.log" 30 8
  awk -F '\t' -v phase="qwen38-trial-${trial_index}-${mode}-settle" '
    $3 != phase { exit 1 }
  ' "$trial_dir/settle.log"

  response="$trial_dir/response.json"
  jq -e '
    .model == "Qwen3.8 27B"
    and (.choices | length) == 1
    and .choices[0].finish_reason == "length"
    and .choices[0].message.role == "assistant"
    and (.choices[0].message.content | type) == "string"
    and (.choices[0].message.content | length) > 0
    and .usage.prompt_tokens >= 100000 and .usage.prompt_tokens <= 120000
    and .usage.completion_tokens == 512
    and .usage.total_tokens == (.usage.prompt_tokens + .usage.completion_tokens)
    and (.x_hf2q_timing.decode_time_secs | type) == "number"
    and .x_hf2q_timing.decode_time_secs > 0
    and (.x_hf2q_timing.decode_tokens_per_sec | type) == "number"
    and .x_hf2q_timing.decode_tokens_per_sec > 0
  ' "$response" >/dev/null
  prompt_tokens=$(jq -er .usage.prompt_tokens "$response")
  completion_tokens=$(jq -er .usage.completion_tokens "$response")
  decode_seconds=$(jq -er .x_hf2q_timing.decode_time_secs "$response")
  decode_tps=$(jq -er .x_hf2q_timing.decode_tokens_per_sec "$response")
  awk -v total="$total_seconds" -v decode="$decode_seconds" \
    'BEGIN { exit !(total >= decode) }'
  test "$prompt_tokens" = "$(jq -er .prompt_tokens "$trial_json")"
  test "$completion_tokens" = "$(jq -er .completion_tokens "$trial_json")"
  float_close "$decode_seconds" "$(jq -er .decode_seconds "$trial_json")" 0.000001
  float_close "$decode_tps" "$(jq -er .decode_tokens_per_second "$trial_json")" 0.000001
  calculated_tps=$(awk -v tokens="$completion_tokens" -v seconds="$decode_seconds" \
    'BEGIN { printf "%.6f", tokens / seconds }')
  float_close "$decode_tps" "$calculated_tps" 0.01

  jq -S '{model,choices,usage}' "$response" >"$tmp/semantic-$trial_index.json"
  cmp "$tmp/semantic-$trial_index.json" "$trial_dir/semantic.json"
  trial_semantic_sha=$(sha256_file "$trial_dir/semantic.json")
  test "$trial_semantic_sha" = "$(jq -er .semantic_sha256 "$trial_json")"
  if [[ -z "$semantic_sha" ]]; then
    semantic_sha=$trial_semantic_sha
  else
    test "$trial_semantic_sha" = "$semantic_sha"
  fi

  server_log="$trial_dir/server.log"
  [[ "$(grep -c 'Qwen35 decode complete' "$server_log")" == 1 ]]
  decode_line=$(grep 'Qwen35 decode complete' "$server_log")
  grep -Eq 'mode=("unary"|unary)' <<<"$decode_line"
  log_generated=$(sed -n 's/.*generated_tokens=\([^ ]*\).*/\1/p' <<<"$decode_line")
  log_elapsed_ms=$(sed -n 's/.*elapsed_ms=\([^ ]*\).*/\1/p' <<<"$decode_line")
  log_tps=$(sed -n 's/.*tokens_per_second=\([^ ]*\).*/\1/p' <<<"$decode_line")
  test "$log_generated" = "$completion_tokens"
  [[ "$log_elapsed_ms" =~ ^[0-9]+([.][0-9]+)?$ ]]
  [[ "$log_tps" =~ ^[0-9]+([.][0-9]+)?$ ]]
  expected_elapsed_ms=$(awk -v seconds="$decode_seconds" 'BEGIN { printf "%.6f", seconds * 1000 }')
  float_close "$log_elapsed_ms" "$expected_elapsed_ms" 5
  float_close "$log_tps" "$decode_tps" 0.2
  if [[ "$mode" == auto ]]; then
    grep -Fq '+ gqa_q2=true ' "$server_log"
    grep -Fq 'Qwen TQ-HB decode selected GQA-cooperative Q2 attention' \
      "$server_log"
    auto_tps+=("$decode_tps")
  else
    grep -Fq '+ gqa_q2=false ' "$server_log"
    if grep -Fq 'Qwen TQ-HB decode selected GQA-cooperative Q2 attention' \
      "$server_log"; then
      echo "Qwen3.8 off-trial evidence contains a GQA Q2 selection" >&2
      exit 1
    fi
    off_tps+=("$decode_tps")
  fi
  if grep -Eiq 'GPU Timeout|SubmissionsIgnored|Command buffer error|Generation error|engine_unhealthy|panicked at|worker-fatal' "$server_log"; then
    echo "Qwen3.8 verifier observed a fatal runtime signature" >&2
    exit 1
  fi
  if grep -Fq 'auto-pipeline: downloading from HF Hub' "$server_log"; then
    echo "Qwen3.8 verifier observed an unsealed auto-pipeline download" >&2
    exit 1
  fi
done

[[ ${#off_tps[@]} == 2 && ${#auto_tps[@]} == 2 ]]
off_median=$(awk -v a="${off_tps[0]}" -v b="${off_tps[1]}" \
  'BEGIN { printf "%.6f", (a+b)/2 }')
auto_median=$(awk -v a="${auto_tps[0]}" -v b="${auto_tps[1]}" \
  'BEGIN { printf "%.6f", (a+b)/2 }')
improvement_percent=$(awk -v baseline="$off_median" -v candidate="$auto_median" \
  'BEGIN { if (baseline <= 0) exit 1; printf "%.6f", ((candidate / baseline) - 1) * 100 }')
float_close "$off_median" \
  "$(jq -er .aggregate.off_median_decode_tokens_per_second "$summary")" 0.000001
float_close "$auto_median" \
  "$(jq -er .aggregate.auto_median_decode_tokens_per_second "$summary")" 0.000001
float_close "$improvement_percent" \
  "$(jq -er .aggregate.improvement_percent "$summary")" 0.000001
test "$semantic_sha" = "$(jq -er .aggregate.exact_output_sha256 "$summary")"
awk -v observed="$improvement_percent" 'BEGIN { exit !(observed >= 15) }'

if [[ "$verification_scope" == release ]]; then
  envelope="$receipt_root/receipt.json"
  thermal_summary="$receipt_root/thermal/summary.json"
  measurement_log="$receipt_root/thermal/measurement.log"
  settle_log="$receipt_root/thermal/settle.log"
  for artifact in "$envelope" "$thermal_summary" "$measurement_log" "$settle_log"; do
    [[ -s "$artifact" ]] || {
      echo "Qwen3.8 release receipt is missing thermal evidence: $artifact" >&2
      exit 1
    }
  done
  jq -e --slurpfile benchmark "$summary" --slurpfile thermal "$thermal_summary" '
    .schema_version == 1 and .status == "pass"
    and .benchmark == $benchmark[0]
    and .thermal == $thermal[0]
  ' "$envelope" >/dev/null
  jq -e --arg benchmark_sha256 "$(sha256_file "$summary")" '
    .status == "pass"
    and .phase == "qwen38-long-decode"
    and .required_start_state == "nominal"
    and .maximum_measurement_state == "fair"
    and .runtime_preflight == "pass"
    and .measurement_scope == "full-abba-benchmark"
    and .benchmark_summary_sha256 == $benchmark_sha256
    and .settle_seconds == 60
    and .settle_duration_seconds >= .settle_seconds
    and .settle_samples > 0
    and .measurement_samples >= 2
    and .measurement_duration_seconds > 0
    and .sample_interval_seconds == 2
    and .maximum_sample_gap_seconds == 5
    and .settle_sample_interval_seconds == 5
    and .maximum_settle_sample_gap_seconds == 8
    and .non_nominal_measurement_samples >= 0
    and .fair_measurement_samples >= 0
    and .over_limit_measurement_samples == 0
    and .settle_telemetry_gaps == 0
    and .telemetry_gaps == 0
    and (.settle_log_sha256 | test("^[0-9a-f]{64}$"))
    and (.measurement_log_sha256 | test("^[0-9a-f]{64}$"))
  ' "$thermal_summary" >/dev/null
  test "$(sha256_file "$measurement_log")" = \
    "$(jq -er .measurement_log_sha256 "$thermal_summary")"
  test "$(sha256_file "$settle_log")" = \
    "$(jq -er .settle_log_sha256 "$thermal_summary")"
  thermal_validate_fair_or_better_measurement_log "$measurement_log" 5
  test "$THERMAL_LOG_SAMPLES" = "$(jq -er .measurement_samples "$thermal_summary")"
  test "$THERMAL_LOG_DURATION_SECONDS" = \
    "$(jq -er .measurement_duration_seconds "$thermal_summary")"
  test "$THERMAL_LOG_NON_NOMINAL_SAMPLES" = \
    "$(jq -er .non_nominal_measurement_samples "$thermal_summary")"
  test "$THERMAL_LOG_FAIR_SAMPLES" = \
    "$(jq -er .fair_measurement_samples "$thermal_summary")"
  test "$THERMAL_LOG_OVER_LIMIT_SAMPLES" = \
    "$(jq -er .over_limit_measurement_samples "$thermal_summary")"
  test "$THERMAL_LOG_GAPS" = "$(jq -er .telemetry_gaps "$thermal_summary")"
  test "$(head -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
    qwen38-long-decode-measurement-start
  test "$(head -1 "$measurement_log" | awk -F '\t' '{print $2}')" = nominal
  test "$(tail -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
    qwen38-long-decode-measurement-end
  awk -F '\t' '
    NR > 1 && $3 != "qwen38-long-decode-measurement" &&
      $3 != "qwen38-long-decode-measurement-end" { exit 1 }
  ' "$measurement_log"

  decode_state_ranks=()
  trial_index=0
  for mode in off auto auto off; do
    trial_index=$((trial_index + 1))
    request_end=$(awk -F '\t' -v trial="$trial_index" -v mode="$mode" '
      $2 == trial && $3 == mode && $4 == "request-end" { print $1 }
    ' "$benchmark_dir/phase.log")
    decode_seconds=$(jq -er .decode_seconds \
      "$benchmark_dir/trial-${trial_index}-${mode}/trial.json")
    decode_start=$(awk -v end="$request_end" -v seconds="$decode_seconds" \
      'BEGIN { print int(end - seconds - 2) }')
    decode_end=$((request_end + 2))
    decode_rank=$(awk -F '\t' -v start="$decode_start" -v end="$decode_end" '
      function rank(state) {
        if (state == "nominal") return 0
        if (state == "fair") return 1
        if (state == "serious") return 2
        if (state == "critical") return 3
        return 4
      }
      $1 >= start && $1 <= end {
        value = rank($2)
        if (value > maximum) maximum = value
        samples++
      }
      END {
        if (samples < 2 || maximum > 1) exit 1
        print maximum
      }
    ' "$measurement_log")
    decode_state_ranks+=("$decode_rank")
  done
  [[ ${#decode_state_ranks[@]} == 4 ]]
  test "${decode_state_ranks[0]}" = "${decode_state_ranks[1]}"
  test "${decode_state_ranks[0]}" = "${decode_state_ranks[2]}"
  test "${decode_state_ranks[0]}" = "${decode_state_ranks[3]}"
  thermal_validate_settle_log "$settle_log" 60 8
  test "$THERMAL_LOG_SAMPLES" = "$(jq -er .settle_samples "$thermal_summary")"
  test "$THERMAL_LOG_DURATION_SECONDS" = \
    "$(jq -er .settle_duration_seconds "$thermal_summary")"
  test "$THERMAL_LOG_GAPS" = "$(jq -er .settle_telemetry_gaps "$thermal_summary")"
  awk -F '\t' '$3 != "qwen38-long-decode-settle" { exit 1 }' "$settle_log"
fi

echo "Qwen3.8 long-decode $verification_scope receipt verified" >&2
