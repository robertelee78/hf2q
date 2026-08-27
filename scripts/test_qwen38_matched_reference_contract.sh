#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runner="$root_dir/scripts/qwen38_matched_reference_abba.sh"
physical_runner="$root_dir/scripts/qwen38_matched_physical_abba.sh"

# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$root_dir/scripts/qwen38_artifact_contract.sh"
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

unmarked_error="$fixture_dir/unmarked-bootstrap.err"
if "$runner" >/dev/null 2>"$unmarked_error"; then
    fail 'unconfigured matched runner unexpectedly succeeded'
fi
grep -Fq 'HF2Q_BIN is required' "$unmarked_error" \
    || fail 'matched runner did not self-bootstrap before environment validation'
marked_error="$fixture_dir/marked-bootstrap.err"
if HF2Q_MATCHED_GATE_ISOLATED=1 "$runner" \
    >/dev/null 2>"$marked_error"; then
    fail 'forced isolation marker unexpectedly succeeded'
fi
grep -Fq 'does not own an isolated process group' "$marked_error" \
    || fail 'forced isolation marker bypassed process-group leadership proof'

expect_failure() {
    local label=$1
    shift
    if "$@" >/dev/null 2>&1; then
        fail "negative matched-reference fixture passed: $label"
    fi
}

# Port availability and calibration sampling are shared fail-closed
# predicates. Exercise their behavior directly so a caller's conditional
# context cannot turn a failed probe into a successful observation.
(
    # Hosted macOS runners do not install ripgrep. Port safety must not become
    # a false success merely because a developer workstation does.
    # shellcheck disable=SC2329
    rg() { return 127; }
    # shellcheck disable=SC2329
    lsof() {
        printf '%s\n' \
          'COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME' \
          'hf2q 99 test 1u IPv4 0 0t0 TCP *:18086 (LISTEN)'
    }
    expect_failure occupied-port matched_require_port_available 18086
)
(
    lsof() { return 1; }
    matched_require_port_available 18086
)

calibration_dir="$fixture_dir/calibration"
mkdir -p "$calibration_dir"
(
    power_mode_code=2
    power_mode_name=automatic
    require_ac_power() { return 0; }
    read_live_power_mode_code() { printf '%s\n' 2; }
    thermal_sample() {
        THERMAL_SAMPLED_AT=101
        THERMAL_STATE=nominal
        printf '101\tnominal\tmeasurement\n' >>"$1"
    }
    host_contention_sample() {
        HOST_CONTENTION_STATE=quiet
        printf '101\tquiet\tmeasurement\t77\t0.0\t-\n' >>"$1"
    }
    matched_record_calibration_observation \
      "$calibration_dir/thermal-success.tsv" \
      "$calibration_dir/host-success.tsv" \
      "$calibration_dir/contention-success.tsv" measurement 77 88
    [[ "$(cat "$calibration_dir/host-success.tsv")" \
      == $'101\tac\tquiet\tautomatic\t2\tmeasurement' ]]
)
for failed_step in ac power-read power-mismatch thermal contention host-write; do
    (
        power_mode_code=2
        power_mode_name=automatic
        require_ac_power() { return 0; }
        read_live_power_mode_code() { printf '%s\n' 2; }
        thermal_sample() {
            THERMAL_SAMPLED_AT=101
            THERMAL_STATE=nominal
            return 0
        }
        host_contention_sample() {
            HOST_CONTENTION_STATE=quiet
            return 0
        }
        case "$failed_step" in
            ac) require_ac_power() { return 91; } ;;
            power-read) read_live_power_mode_code() { return 91; } ;;
            power-mismatch) read_live_power_mode_code() { printf '%s\n' 1; } ;;
            thermal) thermal_sample() { return 91; } ;;
            contention) host_contention_sample() { return 91; } ;;
        esac
        host_path="$calibration_dir/host-$failed_step.tsv"
        [[ "$failed_step" == host-write ]] && host_path="$calibration_dir"
        if matched_record_calibration_observation \
          "$calibration_dir/thermal-$failed_step.tsv" "$host_path" \
          "$calibration_dir/contention-$failed_step.tsv" measurement 77 88; then
            fail "calibration observation masked $failed_step failure"
        fi
    )
done

jq -n \
  --arg hf2q_speculation "$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY" \
  --arg reference_speculation "$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY" \
  --arg hf2q_kv "$QWEN38_MATCHED_HF2Q_KV_CACHE" \
  --arg reference_k "$QWEN38_MATCHED_REFERENCE_KV_CACHE_K" \
  --arg reference_v "$QWEN38_MATCHED_REFERENCE_KV_CACHE_V" \
  --argjson context "$QWEN38_MATCHED_CONTEXT_TOKENS" '{
    schema:2,
    hf2q:{dense_decode_mvn:1,dense_decode_mv_ext:0,
      dense_q5k_canonical_q4x4:1,
      speculation:$hf2q_speculation,kv_cache:$hf2q_kv,
      kv_cache_budget_bytes:51539607552,
      context_tokens_per_slot:$context},
    reference:{speculation:$reference_speculation,kv_cache_k:$reference_k,
      kv_cache_v:$reference_v,context_tokens_total:$context}
  }' >"$fixture_dir/launch-settings.json"
matched_validate_launch_settings "$fixture_dir/launch-settings.json" 1 0 \
  1 \
  "$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY" \
  "$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY"
for mutation in schema mvn mv-ext q5k hf2q-spec reference-spec hf2q-cache hf2q-budget \
    reference-cache context; do
    case "$mutation" in
        schema) filter='.schema=1' ;;
        mvn) filter='.hf2q.dense_decode_mvn=0' ;;
        mv-ext) filter='.hf2q.dense_decode_mv_ext=1' ;;
        q5k) filter='.hf2q.dense_q5k_canonical_q4x4=0' ;;
        hf2q-spec) filter='.hf2q.speculation="fixed-k3-mtp"' ;;
        reference-spec) filter='.reference.speculation="adaptive-history-then-mtp-cost-gated"' ;;
        hf2q-cache) filter='.hf2q.kv_cache="q8_0"' ;;
        hf2q-budget) filter='.hf2q.kv_cache_budget_bytes += 1' ;;
        reference-cache) filter='.reference.kv_cache_k="tq-kv"' ;;
        context) filter='.reference.context_tokens_total=16384' ;;
    esac
    jq "$filter" "$fixture_dir/launch-settings.json" \
      >"$fixture_dir/launch-settings-$mutation.json"
    expect_failure "launch-settings-$mutation" matched_validate_launch_settings \
      "$fixture_dir/launch-settings-$mutation.json" 1 0 \
      1 \
      "$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY" \
      "$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY"
done

# Requested env is not proof of effective routing. The completed server log
# must contain exactly one well-formed frozen-policy line with all three
# effective values.
printf '%s\n' \
  'INFO frozen Qwen GGML routing policy dense_decode_mvn=true dense_decode_mv_ext=false dense_q5k_canonical_q4x4=true' \
  >"$fixture_dir/frozen-policy.log"
matched_validate_qwen_frozen_routing_policy_log \
  "$fixture_dir/frozen-policy.log" 1 0 1
for mutation in missing duplicate wrong-mvn wrong-mv-ext wrong-q5k malformed-q5k; do
    case "$mutation" in
        missing) : >"$fixture_dir/frozen-policy-$mutation.log" ;;
        duplicate) printf '%s\n%s\n' \
          'INFO frozen Qwen GGML routing policy dense_decode_mvn=true dense_decode_mv_ext=false dense_q5k_canonical_q4x4=true' \
          'INFO frozen Qwen GGML routing policy dense_decode_mvn=true dense_decode_mv_ext=false dense_q5k_canonical_q4x4=true' \
          >"$fixture_dir/frozen-policy-$mutation.log" ;;
        wrong-mvn) sed 's/dense_decode_mvn=true/dense_decode_mvn=false/' \
          "$fixture_dir/frozen-policy.log" >"$fixture_dir/frozen-policy-$mutation.log" ;;
        wrong-mv-ext) sed 's/dense_decode_mv_ext=false/dense_decode_mv_ext=true/' \
          "$fixture_dir/frozen-policy.log" >"$fixture_dir/frozen-policy-$mutation.log" ;;
        wrong-q5k) sed 's/dense_q5k_canonical_q4x4=true/dense_q5k_canonical_q4x4=false/' \
          "$fixture_dir/frozen-policy.log" >"$fixture_dir/frozen-policy-$mutation.log" ;;
        malformed-q5k) sed 's/dense_q5k_canonical_q4x4=true/dense_q5k_canonical_q4x4=1/' \
          "$fixture_dir/frozen-policy.log" >"$fixture_dir/frozen-policy-$mutation.log" ;;
    esac
    expect_failure "frozen-policy-$mutation" \
      matched_validate_qwen_frozen_routing_policy_log \
      "$fixture_dir/frozen-policy-$mutation.log" 1 0 1
done

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

# Code quality is behavioral, not byte-trajectory equality between engines.
# Each accepted response must stop naturally, yield one complete Rust source
# file, compile, and pass evaluator-owned tests for the requested function.
write_code_response() {
    local path=$1
    local content=$2
    local finish_reason=${3:-stop}
    jq -n --arg content "$content" --arg finish "$finish_reason" '{
      choices:[{message:{role:"assistant",content:$content,tool_calls:null},
        finish_reason:$finish}],
      usage:{prompt_tokens:10,completion_tokens:20}
    }' >"$path"
}

write_code_response "$fixture_dir/code-a.json" $'```rust\nfn fibonacci(n: u64) -> u64 {\n    let (mut a, mut b) = (0, 1);\n    for _ in 0..n { (a, b) = (b, a + b); }\n    a\n}\n#[cfg(test)] mod tests { use super::*; #[test] fn one() { assert_eq!(fibonacci(5), 5); } }\n```'
write_code_response "$fixture_dir/code-b.json" $'fn binary_search(xs: &[i32], needle: i32) -> Option<usize> {\n    let (mut lo, mut hi) = (0, xs.len());\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match xs[mid].cmp(&needle) {\n            std::cmp::Ordering::Less => lo = mid + 1,\n            std::cmp::Ordering::Greater => hi = mid,\n            std::cmp::Ordering::Equal => return Some(mid),\n        }\n    }\n    None\n}\n#[cfg(test)] mod tests { use super::*; #[test] fn one() { assert_eq!(binary_search(&[1, 3], 3), Some(1)); } }'
write_code_response "$fixture_dir/code-c.json" $'fn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 { let remainder = a % b; a = b; b = remainder; }\n    a\n}\n#[cfg(test)] mod tests { use super::*; #[test] fn one() { assert_eq!(gcd(8, 6), 2); } }'
for name in code-a code-b code-c; do
    matched_extract_rust_source "$fixture_dir/$name.json" \
      "$fixture_dir/$name.rs"
    matched_validate_rust_case "$name" "$fixture_dir/$name.rs" \
      "$fixture_dir/code-validation"
    jq -e '.complete_rust and .compiled and .model_unit_test_present
      and .model_assertion_count == 1
      and (.evaluator_test | startswith("hf2q_contract_tests::evaluator_"))
      and .evaluator_tests_passed' \
      "$fixture_dir/code-validation/$name-quality.json" >/dev/null
done
write_code_response "$fixture_dir/code-length.json" \
  'fn fibonacci(n: u64) -> u64 { n }' length
expect_failure truncated-code-response matched_extract_rust_source \
  "$fixture_dir/code-length.json" "$fixture_dir/code-length.rs"
write_code_response "$fixture_dir/code-wrong.json" \
  $'fn gcd(_a: u64, _b: u64) -> u64 { 1 }\n#[cfg(test)] mod tests { use super::*; #[test] fn one() { assert_eq!(gcd(1, 1), 1); } }'
matched_extract_rust_source "$fixture_dir/code-wrong.json" \
  "$fixture_dir/code-wrong.rs"
expect_failure wrong-code-behavior matched_validate_rust_case code-c \
  "$fixture_dir/code-wrong.rs" "$fixture_dir/wrong-validation"
printf '%s\n' \
  'fn gcd(a: u64, _b: u64) -> u64 { a }' \
  '#[test] fn one() { assert_eq!(gcd(1, 1), 1); assert_ne!(gcd(2, 1), 0); }' \
  >"$fixture_dir/code-multiple-assertions.rs"
expect_failure multiple-model-assertions matched_validate_rust_case code-c \
  "$fixture_dir/code-multiple-assertions.rs" \
  "$fixture_dir/multiple-assertions-validation"
# This authored test would terminate an unfiltered test process successfully.
# Exact evaluator selection must bypass it and expose the wrong implementation.
printf '%s\n' \
  'fn fibonacci(n: u64) -> u64 { n }' \
  '#[test] fn malicious() { std::process::exit(0); assert_eq!(fibonacci(1), 1); }' \
  >"$fixture_dir/code-exit-zero.rs"
expect_failure authored-exit-zero matched_validate_rust_case code-a \
  "$fixture_dir/code-exit-zero.rs" "$fixture_dir/exit-zero-validation"

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
jq '.choices[0].message.reasoning_content="hidden"' \
  "$fixture_dir/response-good.json" >"$fixture_dir/response-reasoning.json"
expect_failure unwanted-reasoning matched_validate_common_response \
  "$fixture_dir/response-reasoning.json"
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

# Calibration accepts only aligned nominal/ac/quiet/full-power observations
# with bounded gaps and explicit measurement boundary markers.
printf '%s\n' \
  $'100\tnominal\tmeasurement-start' \
  $'102\tnominal\tmeasurement' \
  $'104\tnominal\tmeasurement-end' \
  >"$fixture_dir/thermal-good.tsv"
printf '%s\n' \
  $'100\tac\tquiet\tautomatic\t0\tmeasurement-start' \
  $'102\tac\tquiet\tautomatic\t0\tmeasurement' \
  $'104\tac\tquiet\tautomatic\t0\tmeasurement-end' \
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
sed 's/\tautomatic\t0\t/\tlow\t2\t/' "$fixture_dir/host-good.tsv" \
  >"$fixture_dir/host-low-power.tsv"
expect_failure low-power-calibration matched_validate_host_observation_log \
  "$fixture_dir/host-low-power.tsv" 3 4 3
sed '2s/\tautomatic\t0\t/\thigh\t2\t/' "$fixture_dir/host-good.tsv" \
  >"$fixture_dir/host-mode-change.tsv"
expect_failure changing-power-mode matched_validate_host_observation_log \
  "$fixture_dir/host-mode-change.tsv" 3 4 3
printf '%s\n' $'100\tnominal\tmeasurement-start' \
  $'104\tnominal\tmeasurement-end' >"$fixture_dir/thermal-gap.tsv"
expect_failure thermal-sampling-gap thermal_validate_measurement_log \
  "$fixture_dir/thermal-gap.tsv" 3
printf '%s\n' $'100\tac\tquiet\tautomatic\t0\tmeasurement-start' \
  $'104\tac\tquiet\tautomatic\t0\tmeasurement-end' \
  >"$fixture_dir/host-gap.tsv"
expect_failure host-sampling-gap matched_validate_host_observation_log \
  "$fixture_dir/host-gap.tsv" 2 4 3

cat >"$fixture_dir/power-automatic.txt" <<'EOF'
    System Power Settings:
      AC Power:
          Current Power Source: Yes
          High Power Mode: No
          Low Power Mode: No
      Battery Power:
          High Power Mode: No
          Low Power Mode: Yes
EOF
[[ "$(matched_parse_ac_power_mode <"$fixture_dir/power-automatic.txt")" \
  == automatic ]]
sed 's/High Power Mode: No/High Power Mode: Yes/; s/Low Power Mode: No/Low Power Mode: No/' \
  "$fixture_dir/power-automatic.txt" >"$fixture_dir/power-high.txt"
[[ "$(matched_parse_ac_power_mode <"$fixture_dir/power-high.txt")" == high ]]
cat >"$fixture_dir/power-low.txt" <<'EOF'
    System Power Settings:
      AC Power:
          Current Power Source: Yes
          High Power Mode: No
          Low Power Mode: Yes
      Battery Power:
          High Power Mode: No
          Low Power Mode: Yes
EOF
[[ "$(matched_parse_ac_power_mode <"$fixture_dir/power-low.txt")" == low ]]
printf '%s\n' 'System-wide power settings:' 'Currently in use:' \
  ' powermode            2' ' womp                 1' \
  >"$fixture_dir/power-live.txt"
[[ "$(matched_parse_live_power_mode_code <"$fixture_dir/power-live.txt")" == 2 ]]
expect_failure missing-live-power-mode matched_parse_live_power_mode_code \
  < /dev/null
printf '%s\n' ' powermode nope' >"$fixture_dir/power-live-invalid.txt"
expect_failure invalid-live-power-mode matched_parse_live_power_mode_code \
  <"$fixture_dir/power-live-invalid.txt"
printf '%s\n' ' powermode 0' ' powermode 2' \
  >"$fixture_dir/power-live-duplicate.txt"
expect_failure duplicate-live-power-mode matched_parse_live_power_mode_code \
  <"$fixture_dir/power-live-duplicate.txt"
printf '%s\n' "Now drawing from 'AC Power'" \
  ' -InternalBattery-0 (id=1) 100%; charged; 0:00 remaining present: true' \
  >"$fixture_dir/power-source-ac.txt"
[[ "$(matched_parse_live_power_source <"$fixture_dir/power-source-ac.txt")" == ac ]]
printf '%s\n' "Now drawing from 'Battery Power'" \
  >"$fixture_dir/power-source-battery.txt"
[[ "$(matched_parse_live_power_source <"$fixture_dir/power-source-battery.txt")" \
  == battery ]]
expect_failure missing-live-power-source matched_parse_live_power_source \
  < /dev/null
printf '%s\n' "Now drawing from 'AC Power'" "Now drawing from 'Battery Power'" \
  >"$fixture_dir/power-source-duplicate.txt"
expect_failure duplicate-live-power-source matched_parse_live_power_source \
  <"$fixture_dir/power-source-duplicate.txt"

write_stability_fixture() {
    local path=$1
    local hf2q_first=$2
    local hf2q_last=$3
    local reference_first=$4
    local reference_last=$5
    local engine trial factor name group case_index wall tps tokens decode_seconds
    : >"$path"
    for engine in hf2q reference; do
        for trial in $(if [[ "$engine" == hf2q ]]; then printf '1 4'; else printf '2 3'; fi); do
            if [[ "$engine/$trial" == hf2q/1 ]]; then factor=$hf2q_first
            elif [[ "$engine/$trial" == hf2q/4 ]]; then factor=$hf2q_last
            elif [[ "$engine/$trial" == reference/2 ]]; then factor=$reference_first
            else factor=$reference_last
            fi
            case_index=0
            for name in code-a code-b code-c repeat-a repeat-b repeat-c; do
                case_index=$((case_index + 1))
                group=${name%%-*}
                wall=$(awk -v base="$case_index" -v factor="$factor" \
                  'BEGIN { printf "%.9f", base * factor }')
                tps=$factor
                tokens=$((40 + case_index))
                decode_seconds=$(awk -v tokens="$tokens" -v tps="$tps" \
                  'BEGIN { printf "%.9f", tokens / tps }')
                jq -cn --arg engine "$engine" --arg name "$name" \
                  --arg group "$group" --argjson trial "$trial" \
                  --argjson wall "$wall" --argjson tps "$tps" \
                  --argjson decode_seconds "$decode_seconds" \
                  --argjson tokens "$tokens" \
                  '{engine:$engine,trial:$trial,name:$name,group:$group,
                    wall_seconds:$wall,internal_decode_tps:$tps,
                    internal_decode_seconds:$decode_seconds,
                    completion_tokens:$tokens}' >>"$path"
            done
        done
    done
}

validate_stability_fixture() {
    matched_measurement_stability_json "$1" 5 10 | jq -e '.stable == true' \
      >/dev/null
}

# The exact ABBA shape passes at stable frequency, including the 5% boundary.
write_stability_fixture "$fixture_dir/stability-good.jsonl" 0.975 1.025 1 1
validate_stability_fixture "$fixture_dir/stability-good.jsonl"
matched_measurement_stability_json "$fixture_dir/stability-good.jsonl" 5 10 \
  | jq -e '.observed_band_dominance == false' >/dev/null
write_stability_fixture "$fixture_dir/stability-dominant.jsonl" 0.80 0.81 1 1
matched_measurement_stability_json "$fixture_dir/stability-dominant.jsonl" 5 10 \
  | jq -e '.stable == true and .observed_band_dominance == true' >/dev/null

# Nominal host telemetry cannot hide a same-engine 50 -> 28 t/s collapse.
write_stability_fixture "$fixture_dir/stability-collapse.jsonl" 1 1.78 1 1
expect_failure dvfs-collapse validate_stability_fixture \
  "$fixture_dir/stability-collapse.jsonl"

write_stability_fixture "$fixture_dir/stability-over-limit.jsonl" 0.974 1.026 1 1
expect_failure over-five-percent validate_stability_fixture \
  "$fixture_dir/stability-over-limit.jsonl"
write_stability_fixture "$fixture_dir/stability-case-base.jsonl" 1 1 1 1
jq -c 'if .engine == "hf2q" and .trial == 4 and .name == "code-a"
  then .wall_seconds = 1.1052631578947367
    | .internal_decode_tps = 1.1052631578947367
    | .internal_decode_seconds = (.completion_tokens / .internal_decode_tps)
  else . end' "$fixture_dir/stability-case-base.jsonl" \
  >"$fixture_dir/stability-case-boundary.jsonl"
validate_stability_fixture "$fixture_dir/stability-case-boundary.jsonl"
jq -c 'if .engine == "hf2q" and .trial == 4 and .name == "code-a"
  then .wall_seconds = 1.106
    | .internal_decode_tps = 1.106
    | .internal_decode_seconds = (.completion_tokens / .internal_decode_tps)
  else . end' "$fixture_dir/stability-case-base.jsonl" \
  >"$fixture_dir/stability-case-over.jsonl"
expect_failure over-ten-percent-case validate_stability_fixture \
  "$fixture_dir/stability-case-over.jsonl"
sed '1d' "$fixture_dir/stability-good.jsonl" \
  >"$fixture_dir/stability-missing.jsonl"
expect_failure missing-abba-row validate_stability_fixture \
  "$fixture_dir/stability-missing.jsonl"
jq 'if .engine == "hf2q" and .trial == 4 and .name == "code-a"
  then .completion_tokens += 1 else . end' \
  "$fixture_dir/stability-good.jsonl" >"$fixture_dir/stability-token-drift.jsonl"
expect_failure completion-token-drift validate_stability_fixture \
  "$fixture_dir/stability-token-drift.jsonl"

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

# Killing only a matrix-owned nested supervisor must reap its isolated leaf
# and the leaf's long-lived model stand-in.
nested_pid_file="$fixture_dir/nested-supervisor.pids"
"$root_dir/scripts/run_release_gate_process_group.sh" bash -c '
  sleep 300 & grandchild=$!
  printf "%s %s\n" "$$" "$grandchild" >"$1"
  wait "$grandchild"
' _ "$nested_pid_file" &
nested_supervisor=$!
deadline=$((SECONDS + 10))
while [[ ! -s "$nested_pid_file" && $SECONDS -lt $deadline ]]; do sleep 1; done
[[ -s "$nested_pid_file" ]] || fail 'nested supervisor fixture did not start'
read -r nested_leaf nested_grandchild <"$nested_pid_file"
matched_terminate_owned_child "$nested_supervisor"
for reaped_pid in "$nested_supervisor" "$nested_leaf" "$nested_grandchild"; do
    if kill -0 "$reaped_pid" 2>/dev/null; then
        fail "nested supervisor cancellation leaked PID $reaped_pid"
    fi
done

# The production runner must call, rather than merely mention, each tested
# predicate. A caller-provided model ID is initialized before any trial.
[[ "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" == 1 ]] \
  || fail 'matched Q5_K policy is not uniformly enabled'
grep -Fq 'env -i' "$runner" \
  || fail 'matched runner does not launch from a clean environment'
# shellcheck disable=SC2016
for call in \
  'matched_resolve_hf2q_model_id ' \
  'matched_validate_reference_model_alias ' \
  'matched_validate_qwen_frozen_routing_policy_log "$log" ' \
  '| matched_parse_sse_stream ' \
  'matched_extract_rust_source "$response" "$source_path"' \
  'matched_validate_rust_case "$name" ' \
  'matched_validate_hf2q_telemetry "$response"' \
  'matched_validate_reference_telemetry "$response"' \
  'matched_reference_speculation_totals "$rows_file"' \
  'matched_validate_host_observation_log ' \
  'matched_parse_ac_power_mode' \
  'matched_parse_live_power_mode_code' \
  'matched_measurement_stability_json "$rows_file"' \
  'hf2q_macos_verify_runtime_manifest "$REFERENCE_BIN"' \
  'matched_publish_result ' \
  'matched_validate_reopened_reference_child "$OUT_DIR"'; do
    grep -Fq "$call" "$runner" \
      || fail "production runner does not invoke tested predicate: $call"
done
for contention_call in \
  'host_contention_sample ' \
  'host_contention_require_quiet ' \
  'host_contention_validate_settle_log ' \
  'host_contention_validate_measurement_log ' \
  'host_contention_validate_thermal_alignment '; do
  grep -Fq "$contention_call" "$runner" \
    || fail "matched runner omits v2 contention authority: $contention_call"
done
grep -Fq '"$THERMAL_SAMPLED_AT" "$owned_server_pid"' \
  "$root_dir/scripts/qwen38_matched_reference_contract.sh" \
  || fail 'shared calibration predicate does not narrowly exempt its owned server PID'
if grep -Fq 'require_no_foreign_heavy_work ' "$runner"; then
  fail 'matched runner still uses the name-only contention predicate'
fi
grep -Fq 'matched_require_port_available "$PORT"' "$physical_runner" \
  || fail 'matched physical runner bypasses the tested port predicate'
for calibrated_runner in "$runner" "$physical_runner"; do
  grep -Fq 'matched_record_calibration_observation "$1" "$2" "$3" "$4"' \
    "$calibrated_runner" \
    || fail 'matched runner bypasses the tested calibration predicate'
done
grep -Fq 'policy:$host_contention_policy' "$runner" \
  || fail 'matched summary omits its exact contention policy'
grep -Fq 'HOST_CONTENTION_GATE_OWNER_PID' "$runner" \
  || fail 'matched runner does not use one stable process-group owner'
grep -Fq 'matched_terminate_owned_child "$child_pid"' \
  "$root_dir/scripts/qwen38_matched_peer_matrix.sh" \
  || fail 'matched peer matrix does not own nested leaf cleanup'
grep -Fq "trap 'exit 130' INT TERM" \
  "$root_dir/scripts/qwen38_matched_peer_matrix.sh" \
  || fail 'matched peer matrix does not trap targeted cancellation'
grep -Fq 'readonly MAX_TOKENS=256' "$runner" \
  || fail "matched code workload is not sized for complete source"
grep -Fq 'quality_scope:"complete Rust compilation plus evaluator tests; exact repeat transcription"' \
  "$runner" || fail "matched summary lost its executable quality contract"
grep -Fq -- "--argjson dense_decode_mvn \"\$MATCHED_HF2Q_DECODE_MVN\"" "$runner" \
  || fail 'launch receipt does not consume the actual MVN setting'
grep -Fq -- "--argjson dense_decode_mv_ext \"\$MATCHED_HF2Q_DECODE_MV_EXT\"" \
  "$runner" \
  || fail 'launch receipt does not consume the actual MV_EXT setting'
grep -Fq -- '--argjson dense_q5k_canonical_q4x4' "$runner" \
  || fail 'launch receipt does not consume the actual Q5_K setting'
grep -Fq "HF2Q_DECODE_MVN=\"\$MATCHED_HF2Q_DECODE_MVN\"" "$runner" \
  || fail 'hf2q launch does not consume the receipt MVN setting'
grep -Fq "HF2Q_DECODE_MV_EXT=\"\$MATCHED_HF2Q_DECODE_MV_EXT\"" "$runner" \
  || fail 'hf2q launch does not consume the receipt MV_EXT setting'
# shellcheck disable=SC2016
grep -Fq 'HF2Q_Q5K_CANONICAL_Q4X4="$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4"' \
  "$runner" || fail 'hf2q launch does not consume the receipt Q5_K setting'
grep -Fq \
  "KV_CACHE_BUDGET_BYTES=\"\$MATCHED_HF2Q_KV_CACHE_BUDGET_BYTES\"" "$runner" \
  || fail 'hf2q launch does not consume the receipt KV budget'
grep -Fq -- "--ctx-size \"\$QWEN38_MATCHED_CONTEXT_TOKENS\"" "$runner" \
  || fail 'reference launch does not consume the receipt context setting'
grep -Fq -- \
  "--cache-type-k \"\$QWEN38_MATCHED_REFERENCE_KV_CACHE_K\"" "$runner" \
  || fail 'reference launch does not consume the receipt K-cache setting'
grep -Fq -- \
  "--cache-type-v \"\$QWEN38_MATCHED_REFERENCE_KV_CACHE_V\"" "$runner" \
  || fail 'reference launch does not consume the receipt V-cache setting'
grep -Fq "matched_validate_launch_settings \"\$launch_settings\"" "$runner" \
  || fail 'production runner does not fail closed on launch settings'
grep -Fq "runtime_manifest_sha256:\$reference_runtime_manifest_sha" "$runner" \
  || fail "matched summary does not bind the reference runtime closure"
# shellcheck disable=SC2016
grep -Fq 'effective_routing_policy:{dense_decode_mvn:$dense_decode_mvn' \
  "$runner" || fail 'matched summary omits the effective Qwen routing policy'
# shellcheck disable=SC2016
model_id_init_line=$(grep -nE '^if \[\[ -n "\$MODEL_ID" \]\]; then$' "$runner" \
  | cut -d: -f1)
# shellcheck disable=SC2016
trial_loop_line=$(grep -nE '^for engine in \$TRIAL_ORDER; do$' "$runner" | cut -d: -f1)
[[ "$model_id_init_line" =~ ^[0-9]+$ && "$trial_loop_line" =~ ^[0-9]+$
  && model_id_init_line -lt trial_loop_line ]] \
  || fail "caller-provided MODEL_ID is not initialized before trials"
stop_line=$(grep -nE '^[[:space:]]+stop_server$' "$runner" | tail -1 | cut -d: -f1)
# shellcheck disable=SC2016
policy_line=$(grep -n 'matched_validate_qwen_frozen_routing_policy_log "\$log"' \
  "$runner" | cut -d: -f1)
[[ "$stop_line" =~ ^[0-9]+$ && "$policy_line" =~ ^[0-9]+$ \
  && stop_line -lt policy_line ]] \
  || fail 'hf2q effective policy is not re-read from the completed server log'
[[ "$(grep -c '^matched_publish_result ' "$runner")" == 1 ]] \
  || fail "passing summary does not have one audited publication path"
if grep -Eq '^mv .*summary\.json' "$runner"; then
    fail "production runner bypasses summary-last publication"
fi

printf 'qwen38 matched-reference behavioral contract passed\n'
