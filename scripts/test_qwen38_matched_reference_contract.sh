#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runner="$root_dir/scripts/qwen38_matched_reference_abba.sh"

# shellcheck source=scripts/qwen38_matched_reference_contract.sh
source "$root_dir/scripts/qwen38_matched_reference_contract.sh"
# shellcheck source=scripts/macos_thermal_guard.sh
source "$root_dir/scripts/macos_thermal_guard.sh"

fixture_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen38-matched-contract.XXXXXX")
cleanup() {
    case "$fixture_dir" in
        "${TMPDIR:-/tmp}"/hf2q-qwen38-matched-contract.*)
            rm -rf -- "$fixture_dir"
            ;;
        *)
            echo "refusing to remove unexpected fixture path: $fixture_dir" >&2
            return 1
            ;;
    esac
}
trap cleanup EXIT

fail() {
    echo "$*" >&2
    exit 1
}

expect_failure() {
    local label=$1
    shift
    if "$@" >/dev/null 2>&1; then
        fail "negative matched-reference fixture passed: $label"
    fi
}

# Both live /v1/models schemas are parsed behaviorally, including alias and
# cardinality failures.
jq -n '{data:[{id:"hf2q-model",loaded:true},{id:"cold",loaded:false}]}' \
  >"$fixture_dir/hf2q-models.json"
[[ "$(matched_resolve_hf2q_model_id "$fixture_dir/hf2q-models.json")" \
  == hf2q-model ]]
jq -n '{data:[{id:"a",loaded:true},{id:"b",loaded:true}]}' \
  >"$fixture_dir/hf2q-models-multiple.json"
expect_failure hf2q-model-cardinality matched_resolve_hf2q_model_id \
  "$fixture_dir/hf2q-models-multiple.json"

jq -n '{data:[
  {id:"path-id",aliases:["served-model"],status:{value:"loaded"}},
  {id:"cold",aliases:[],status:{value:"unloaded"}}
]}' >"$fixture_dir/reference-models.json"
matched_validate_reference_model_alias \
  "$fixture_dir/reference-models.json" served-model
expect_failure reference-model-alias matched_validate_reference_model_alias \
  "$fixture_dir/reference-models.json" wrong-model

# Current reference servers expose loaded entries directly in `.data` without
# the older nested `status.value`; identity remains exact through id/aliases.
jq -n '{object:"list",data:[{
  id:"Release Qwen38 E2",aliases:["Release Qwen38 E2"],object:"model"
}]}' >"$fixture_dir/reference-models-current.json"
matched_validate_reference_model_alias \
  "$fixture_dir/reference-models-current.json" 'Release Qwen38 E2'
jq -n '{data:[{
  id:"Release Qwen38 E2",aliases:["Release Qwen38 E2"],
  status:{value:"unloaded"}
}]}' >"$fixture_dir/reference-models-unloaded.json"
expect_failure reference-model-unloaded matched_validate_reference_model_alias \
  "$fixture_dir/reference-models-unloaded.json" 'Release Qwen38 E2'

# The streamed TTFT parser must ignore role-only events, require semantic
# content, require exactly one DONE, and compare the complete streamed text.
started_at=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
  -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC) - 0.01')
printf '%s\n' \
  'data: {"choices":[{"delta":{"role":"assistant"}}]}' \
  'data: {"choices":[{"delta":{"content":"STREAM-"}}]}' \
  'data: {"choices":[{"delta":{"content":"TTFT"}}]}' \
  'data: [DONE]' \
  | matched_parse_sse_stream "$started_at" "$fixture_dir/good.sse" \
      "$fixture_dir/good-sse.json" STREAM-TTFT
jq -e '.content == "STREAM-TTFT" and .done_count == 1
  and .event_count == 3 and .first_semantic_ms > 0' \
  "$fixture_dir/good-sse.json" >/dev/null
if printf '%s\n' \
  'data: {"choices":[{"delta":{"role":"assistant"}}]}' \
  'data: [DONE]' \
  | matched_parse_sse_stream "$started_at" "$fixture_dir/role-only.sse" \
      "$fixture_dir/role-only.json" STREAM-TTFT >/dev/null 2>&1; then
    fail "role-only SSE passed the semantic TTFT contract"
fi
if printf '%s\n' \
  'data: {"choices":[{"delta":{"content":"STREAM-TTFT"}}]}' \
  | matched_parse_sse_stream "$started_at" "$fixture_dir/no-done.sse" \
      "$fixture_dir/no-done.json" STREAM-TTFT >/dev/null 2>&1; then
    fail "SSE without DONE passed the TTFT contract"
fi
if printf '%s\n' \
  'data: {"choices":[{"delta":{"content":"different"}}]}' \
  'data: [DONE]' \
  | matched_parse_sse_stream "$started_at" "$fixture_dir/mismatch.sse" \
      "$fixture_dir/mismatch.json" STREAM-TTFT >/dev/null 2>&1; then
    fail "mismatched streamed content passed the TTFT contract"
fi

# Positive runtime telemetry and null/zero counterexamples exercise the same
# predicates called for every measured response.
jq -n '{
  choices:[{message:{role:"assistant",content:"ok",tool_calls:null},
    finish_reason:"stop"}],
  usage:{prompt_tokens:10,completion_tokens:2,
    prompt_tokens_details:{cached_tokens:3}},
  x_hf2q_timing:{prefill_time_secs:0.2,decode_time_secs:0.3,
    time_to_first_token_ms:4,prefill_tokens_per_sec:50,decode_tokens_per_sec:7},
  timings:{cache_n:3,prompt_n:7,predicted_n:2,prompt_ms:200,
    predicted_ms:300,prompt_per_second:50,predicted_per_second:7,
    draft_n:4,draft_n_accepted:2}
}' >"$fixture_dir/response-good.json"
matched_validate_common_response "$fixture_dir/response-good.json"
matched_validate_hf2q_telemetry "$fixture_dir/response-good.json"
matched_validate_reference_telemetry "$fixture_dir/response-good.json"
jq '.usage.prompt_tokens_details.cached_tokens = null
  | .x_hf2q_timing.prefill_time_secs = 0
  | .x_hf2q_timing.decode_time_secs = 0' \
  "$fixture_dir/response-good.json" >"$fixture_dir/response-bad-hf2q.json"
expect_failure hf2q-null-zero-telemetry matched_validate_hf2q_telemetry \
  "$fixture_dir/response-bad-hf2q.json"
jq '.timings.cache_n = null | .timings.prompt_ms = 0
  | .timings.predicted_ms = 0' \
  "$fixture_dir/response-good.json" >"$fixture_dir/response-bad-reference.json"
expect_failure reference-null-zero-telemetry \
  matched_validate_reference_telemetry \
  "$fixture_dir/response-bad-reference.json"

# Each reference trial must independently show both drafted and accepted
# tokens; aggregate activity from one trial cannot cover an inactive peer.
printf '%s\n' \
  '{"engine":"reference","trial":2,"drafted_tokens":4,"accepted_draft_tokens":2}' \
  '{"engine":"reference","trial":2,"drafted_tokens":3,"accepted_draft_tokens":1}' \
  '{"engine":"reference","trial":3,"drafted_tokens":0,"accepted_draft_tokens":0}' \
  >"$fixture_dir/speculation.jsonl"
[[ "$(matched_reference_speculation_totals \
  "$fixture_dir/speculation.jsonl" 2)" == $'7\t3' ]]
expect_failure missing-reference-trial-acceptance \
  matched_reference_speculation_totals "$fixture_dir/speculation.jsonl" 3

# Calibration accepts only aligned nominal/ac/quiet observations with bounded
# gaps and explicit measurement boundary markers.
printf '%s\n' \
  $'100\tnominal\tmeasurement-start' \
  $'102\tnominal\tmeasurement' \
  $'104\tnominal\tmeasurement-end' \
  >"$fixture_dir/thermal-good.tsv"
printf '%s\n' \
  $'100\tac\tquiet\tmeasurement-start' \
  $'102\tac\tquiet\tmeasurement' \
  $'104\tac\tquiet\tmeasurement-end' \
  >"$fixture_dir/host-good.tsv"
thermal_validate_measurement_log "$fixture_dir/thermal-good.tsv" 3
matched_validate_host_observation_log "$fixture_dir/host-good.tsv" 3 4 3
matched_validate_calibration_alignment \
  "$fixture_dir/thermal-good.tsv" "$fixture_dir/host-good.tsv"
sed 's/\tac\t/\tbattery\t/' "$fixture_dir/host-good.tsv" \
  >"$fixture_dir/host-battery.tsv"
expect_failure non-ac-calibration matched_validate_host_observation_log \
  "$fixture_dir/host-battery.tsv" 3 4 3
sed 's/\tquiet\t/\tbusy\t/' "$fixture_dir/host-good.tsv" \
  >"$fixture_dir/host-busy.tsv"
expect_failure busy-calibration matched_validate_host_observation_log \
  "$fixture_dir/host-busy.tsv" 3 4 3
printf '%s\n' $'100\tnominal\tmeasurement-start' \
  $'104\tnominal\tmeasurement-end' >"$fixture_dir/thermal-gap.tsv"
expect_failure thermal-sampling-gap thermal_validate_measurement_log \
  "$fixture_dir/thermal-gap.tsv" 3
printf '%s\n' $'100\tac\tquiet\tmeasurement-start' \
  $'104\tac\tquiet\tmeasurement-end' >"$fixture_dir/host-gap.tsv"
expect_failure host-sampling-gap matched_validate_host_observation_log \
  "$fixture_dir/host-gap.tsv" 2 4 3

# The production host-contention parser must execute under the platform awk,
# exclude the current server PID, and distinguish Python model work from an
# unrelated Python utility. This catches syntax that GNU awk accepts but the
# macOS implementation rejects before a timed trial can start.
scripted_offenders=$(printf '%s\n' \
  '101 /usr/bin/python3 teacher_model_gen.py' \
  '202 /usr/bin/python3 calendar_export.py' \
  '303 /opt/venv/bin/python inference.py' \
  | matched_find_scripted_model_work 303)
[[ "$scripted_offenders" == '101:python-model-work' ]] \
  || fail "portable Python model-work matcher returned: $scripted_offenders"

# A seal-publication failure must not expose summary.json. A successful call
# publishes a self-consistent result manifest and the passing summary last.
prepare_seal_fixture() {
    local output_dir=$1
    mkdir -p "$output_dir"
    printf '%s\n' payload >"$output_dir/payload.txt"
    printf '%s  payload.txt\n' \
      "$(shasum -a 256 "$output_dir/payload.txt" | awk '{print $1}')" \
      >"$output_dir/evidence.sha256"
    printf '%s\n' '{"schema":2,"verdict":"pass"}' \
      >"$output_dir/summary.json.tmp"
}

prepare_seal_fixture "$fixture_dir/seal-failure"
# Called indirectly by matched_publish_result to force its first publication
# step to fail.
# shellcheck disable=SC2329
mv() {
    if [[ ${2:-} == "$fixture_dir/seal-failure/result.sha256" ]]; then
        return 91
    fi
    command mv "$@"
}
expect_failure result-seal-publication matched_publish_result \
  "$fixture_dir/seal-failure/summary.json.tmp" \
  "$fixture_dir/seal-failure/summary.json" \
  "$fixture_dir/seal-failure/evidence.sha256" \
  "$fixture_dir/seal-failure/result.sha256"
unset -f mv
[[ ! -e "$fixture_dir/seal-failure/summary.json" ]] \
  || fail "pass summary remained after result seal publication failed"

prepare_seal_fixture "$fixture_dir/seal-success"
matched_publish_result "$fixture_dir/seal-success/summary.json.tmp" \
  "$fixture_dir/seal-success/summary.json" \
  "$fixture_dir/seal-success/evidence.sha256" \
  "$fixture_dir/seal-success/result.sha256"
(cd "$fixture_dir/seal-success" && shasum -a 256 -c result.sha256 >/dev/null)

# The production runner must call, rather than merely mention, each tested
# predicate. A caller-provided model ID is initialized before any trial.
# shellcheck disable=SC2016
for call in \
  'matched_resolve_hf2q_model_id ' \
  'matched_validate_reference_model_alias ' \
  '| matched_parse_sse_stream ' \
  'matched_validate_hf2q_telemetry "$response"' \
  'matched_validate_reference_telemetry "$response"' \
  'matched_reference_speculation_totals "$rows_file"' \
  'matched_find_scripted_model_work "$allowed_pid"' \
  'matched_validate_host_observation_log ' \
  'matched_publish_result '; do
    grep -Fq "$call" "$runner" \
      || fail "production runner does not invoke tested predicate: $call"
done
# shellcheck disable=SC2016
model_id_init_line=$(grep -nE '^if \[\[ -n "\$MODEL_ID" \]\]; then$' "$runner" \
  | cut -d: -f1)
# shellcheck disable=SC2016
trial_loop_line=$(grep -nE '^for engine in \$TRIAL_ORDER; do$' "$runner" | cut -d: -f1)
[[ "$model_id_init_line" =~ ^[0-9]+$ && "$trial_loop_line" =~ ^[0-9]+$
  && model_id_init_line -lt trial_loop_line ]] \
  || fail "caller-provided MODEL_ID is not initialized before trials"
[[ "$(grep -c '^matched_publish_result ' "$runner")" == 1 ]] \
  || fail "passing summary does not have one audited publication path"
if grep -Eq '^mv .*summary\.json' "$runner"; then
    fail "production runner bypasses summary-last publication"
fi

printf 'qwen38 matched-reference behavioral contract passed\n'
