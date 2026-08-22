#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen38_physical_multislot_contract.sh
source "$script_dir/qwen38_physical_multislot_contract.sh"

for command in awk bash jq mktemp sort; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done

test_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen38-physical-contract.XXXXXX")
cleanup() {
    case "$test_dir" in
        "${TMPDIR:-/tmp}"/hf2q-qwen38-physical-contract.*)
            rm -rf -- "$test_dir"
            ;;
        *)
            echo "refusing to remove unexpected test fixture: $test_dir" >&2
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
        fail "negative physical-width fixture passed: $label"
    fi
}

write_metrics() {
    local path=$1
    local width=$2
    local offset=$3
    printf '%s\n' \
        "hf2q_qwen_decode_scheduler_steps_total $((offset + 10))" \
        "hf2q_qwen_decode_scheduler_handles_total $((offset + 40))" \
        "hf2q_qwen_decode_scheduler_max_width $width" \
        "hf2q_qwen_decode_ordinary_target_forwards_total $((offset + 10))" \
        "hf2q_qwen_decode_ordinary_target_body_rows_total $((offset + 40))" \
        "hf2q_qwen_decode_ordinary_target_body_max_width $width" \
        "hf2q_qwen_decode_ordinary_target_head_rows_total $((offset + 40))" \
        "hf2q_qwen_decode_ordinary_target_head_max_width $width" \
        "hf2q_qwen_decode_ordinary_command_buffers_created_total $((offset + 30))" \
        "hf2q_qwen_decode_ordinary_command_buffer_submissions_total $((offset + 20))" \
        >"$path"
}

write_response() {
    local path=$1
    local prompt_tokens=${2:-32}
    jq -n --argjson prompt_tokens "$prompt_tokens" '{
      id:"fixture",object:"chat.completion",model:"fixture-model",
      choices:[{index:0,message:{role:"assistant",content:"1: cobalt"},
        finish_reason:"length"}],
      usage:{prompt_tokens:$prompt_tokens,completion_tokens:8,
        total_tokens:($prompt_tokens+8),
        prompt_tokens_details:{cached_tokens:0}},
      x_hf2q_timing:{decode_time_secs:1,total_time_secs:2,
        time_to_first_token_ms:100,decode_tokens_per_sec:8}
    }' >"$path"
}

before="$test_dir/before.txt"
after="$test_dir/after.txt"
write_metrics "$before" 1 0
write_metrics "$after" 4 100
qwen38_physical_validate_metrics 4 "$before" "$after"

sed 's/scheduler_max_width 4/scheduler_max_width 2/' "$after" \
    >"$test_dir/wrong-scheduler.txt"
expect_failure scheduler-width qwen38_physical_validate_metrics \
    4 "$before" "$test_dir/wrong-scheduler.txt"
sed 's/target_body_max_width 4/target_body_max_width 2/' "$after" \
    >"$test_dir/wrong-body.txt"
expect_failure body-width qwen38_physical_validate_metrics \
    4 "$before" "$test_dir/wrong-body.txt"
sed 's/target_head_max_width 4/target_head_max_width 2/' "$after" \
    >"$test_dir/wrong-head.txt"
expect_failure head-width qwen38_physical_validate_metrics \
    4 "$before" "$test_dir/wrong-head.txt"
awk '
  /^hf2q_qwen_decode_ordinary_command_buffer_submissions_total / {
    print "hf2q_qwen_decode_ordinary_command_buffer_submissions_total 20"; next
  }
  { print }
' "$after" >"$test_dir/no-submission.txt"
expect_failure no-submission qwen38_physical_validate_metrics \
    4 "$before" "$test_dir/no-submission.txt"
# Allocation/creation activity is recorded but is not mistaken for physical
# submission proof. A zero creation delta is valid when committed submissions
# and the physical row-width counters prove the wave executed.
awk '
  /^hf2q_qwen_decode_ordinary_command_buffers_created_total / {
    print "hf2q_qwen_decode_ordinary_command_buffers_created_total 30"; next
  }
  { print }
' "$after" >"$test_dir/reused-created-buffer.txt"
qwen38_physical_validate_metrics \
    4 "$before" "$test_dir/reused-created-buffer.txt"

for index in 1 2 3 4; do
    write_response "$test_dir/response-$index.json"
    qwen38_physical_validate_response \
        "$test_dir/response-$index.json" fixture-model
done
qwen38_physical_validate_equal_prompt_tokens 4 \
    "$test_dir"/response-{1,2,3,4}.json
jq '.usage.prompt_tokens = 33 | .usage.total_tokens = 41' \
    "$test_dir/response-4.json" >"$test_dir/response-mismatched.json"
expect_failure unequal-prompt-tokens qwen38_physical_validate_equal_prompt_tokens 4 \
    "$test_dir/response-1.json" "$test_dir/response-2.json" \
    "$test_dir/response-3.json" "$test_dir/response-mismatched.json"
jq 'del(.usage.prompt_tokens_details.cached_tokens)' \
    "$test_dir/response-1.json" >"$test_dir/response-no-cache.json"
expect_failure missing-cached-token-count qwen38_physical_validate_response \
    "$test_dir/response-no-cache.json" fixture-model
jq '.choices[0].message.content = ""' \
    "$test_dir/response-1.json" >"$test_dir/response-empty.json"
expect_failure empty-output qwen38_physical_validate_response \
    "$test_dir/response-empty.json" fixture-model

bash -n "$script_dir/qwen38_physical_multislot_gate.sh"
bash -n "$script_dir/qwen38_physical_multislot_contract.sh"
bash -n "$script_dir/serve_qwen36_opencode.sh"
bash -n "$script_dir/serve_qwen38_opencode.sh"
grep -Fq 'MAX_SLOTS > 16' "$script_dir/serve_qwen36_opencode.sh" || \
    fail "canonical Qwen launcher does not permit the N=16 gate"
grep -Fq 'readonly WIDTHS=(1 2 4 8 16)' \
    "$script_dir/qwen38_physical_multislot_gate.sh" || \
    fail "physical-width runner does not require the complete width set"

echo "Qwen3.8 physical multi-slot gate contract: PASS"
