#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/agentic_cache_lifecycle_contract.sh
source "$script_dir/agentic_cache_lifecycle_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"
receipt=${1:?receipt required}
source_root=${2:?source root required}
[[ "$receipt" == /* && -f "$receipt" && ! -L "$receipt" \
    && "$(basename "$receipt")" == receipt.json ]] || exit 2
root=$(cd "$(dirname "$receipt")" && pwd -P)
fail() { echo "invalid Qwen lifecycle cell: $*" >&2; exit 1; }
tmp_dir=$(mktemp -d "${TMPDIR:-/var/tmp}/qwen-lifecycle-verify.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT
jq -e '
  .schema == 2 and .verdict == "pass"
  and .gate == "qwen35-agentic-lifecycle-cell"
  and (.source.commit | test("^[0-9a-f]{40}$"))
  and (.source.sha256 | test("^[0-9a-f]{64}$"))
  and (.model.sha256 | test("^[0-9a-f]{64}$"))
  and (.model.shape == "qwen38-dense" or .model.shape == "qwen36-moe")
  and .model.arch_family == "qwen35"
  and (.runtime | del(.routing)) == {clean_environment:true,max_slots:4,
    scheduler:"inflight-batched",speculation:"auto",
    kv_cache_budget_bytes:51539607552,kv_persist:false,
    cache_dir:"evidence-local"}
  and (.runtime.routing.dense_q5k_canonical_q4x4 | type) == "boolean"
  and .lifecycle.context_lines == 2800
  and .lifecycle.continuation_thinking_token_budget == 16
  and .lifecycle.unrelated_conversation_thinking_enabled == false
  and (.lifecycle.summary_sha256 | test("^[0-9a-f]{64}$"))
  and (.evidence.manifest_sha256 | test("^[0-9a-f]{64}$"))
' "$receipt" >/dev/null || fail contract
source_commit=$(jq -er .source.commit "$receipt")
binary=$(jq -er .source.binary "$receipt")
binary_sha=$(jq -er .source.sha256 "$receipt")
model=$(jq -er .model.path "$receipt")
model_sha=$(jq -er .model.sha256 "$receipt")
model_bytes=$(jq -er .model.bytes "$receipt")
architecture=$(jq -er .model.architecture "$receipt")
model_shape=$(jq -er .model.shape "$receipt")
q5k_canonical_q4x4=$(jq -er .runtime.routing.dense_q5k_canonical_q4x4 "$receipt")
run_id=$(jq -er .lifecycle.run_id "$receipt")
case "$model_shape" in
    qwen38-dense) [[ "$architecture" == qwen35 ]] || fail shape-architecture ;;
    qwen36-moe) [[ "$architecture" == qwen35moe ]] || fail shape-architecture ;;
    *) fail model-shape ;;
esac
[[ "$source_root" == /* \
    && "$(git -C "$source_root" rev-parse HEAD)" == "$source_commit" \
    && -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" \
    && -x "$binary" && "$(shasum -a 256 "$binary" | awk '{print $1}')" == "$binary_sha" \
    && -f "$model" && ! -L "$model" \
    && "$(stat -f '%z' "$model" 2>/dev/null || stat -c '%s' "$model")" == "$model_bytes" ]] \
    || fail identity
grep -aFq "$source_commit" "$binary" || fail binary-commit
[[ "$(jq -er .sha256 "$root/model-verification.json")" == "$model_sha" \
    && "$(jq -er .content_hash_verified "$root/model-verification.json")" == true ]] \
    || fail model-verification
hf2q_release_verify_model "$model" "$model_sha" "$root/model-verification.json" \
    || fail model-snapshot
[[ "$(shasum -a 256 "$root/evidence.sha256" | awk '{print $1}')" == \
      "$(jq -er .evidence.manifest_sha256 "$receipt")" ]] || fail manifest-hash
(cd "$root" && shasum -a 256 -c evidence.sha256 >/dev/null) \
    || fail raw-manifest
[[ "$(shasum -a 256 "$root/lifecycle/summary.json" | awk '{print $1}')" == \
      "$(jq -er .lifecycle.summary_sha256 "$receipt")" ]] || fail summary-hash
agentic_lifecycle_validate_summary "$root/lifecycle/summary.json" "$run_id" 2800 \
    "$model_sha" qwen35 "$architecture" 16 false || fail lifecycle-summary
for phase_stream in base:false seed:false active_sse:true sibling:false isolation:false; do
    phase=${phase_stream%%:*}
    stream=${phase_stream#*:}
    agentic_lifecycle_execution_receipt_json \
        "$root/lifecycle/$phase.response.headers" \
        "$model_sha" qwen35 "$architecture" \
      | jq --arg phase "$phase" --argjson stream "$stream" \
          '. + {phase:$phase,stream:$stream}' >"$tmp_dir/$phase.execution.json" \
      || fail "$phase execution headers"
    cmp -s "$tmp_dir/$phase.execution.json" \
        "$root/lifecycle/$phase.execution.json" \
        || fail "$phase execution receipt derivation"
done
jq -s . "$tmp_dir"/*.execution.json >"$tmp_dir/execution-unsorted.json"
jq '[.[]] | sort_by(
  if .phase == "base" then 0 elif .phase == "seed" then 1
  elif .phase == "active_sse" then 2 elif .phase == "sibling" then 3 else 4 end
)' "$tmp_dir/execution-unsorted.json" >"$tmp_dir/execution.json"
jq -e --slurpfile execution "$tmp_dir/execution.json" \
    '.execution_receipts == $execution[0]' "$root/lifecycle/summary.json" >/dev/null \
    || fail summary-execution-derivation
jq -e --arg run "$run_id" '
  (.choices | length) == 1 and .choices[0].finish_reason == "tool_calls"
  and ((.choices[0].message.tool_calls // []) | length) == 1
  and .choices[0].message.tool_calls[0].function.name == "lifecycle_probe"
  and ((.choices[0].message.tool_calls[0].function.arguments | fromjson).nonce == $run)
  and (.usage.prompt_tokens_details.cached_tokens // 0) == 0
' "$root/lifecycle/base.response.json" >/dev/null || fail raw-base-tool-call
jq -e -s '
  (.[0] | has("thinking_token_budget") | not)
  and all(.[1:4][]; .thinking_token_budget == 16)
  and (.[4] | has("thinking_token_budget") | not)
  and .[4].hf2q_enable_thinking == false
  and (.[4] | has("chat_template_kwargs") | not)
' "$root/lifecycle/base.request.json" \
  "$root/lifecycle/seed.request.json" \
  "$root/lifecycle/active.request.json" \
  "$root/lifecycle/sibling.request.json" \
  "$root/lifecycle/isolation.request.json" >/dev/null \
  || fail raw-thinking-budget-contract
base_prompt=$(jq -er '.usage.prompt_tokens' "$root/lifecycle/base.response.json")
minimum_cached=$((base_prompt > 64 ? base_prompt - 64 : 1))
base_tool_id=$(jq -er '.choices[0].message.tool_calls[0].id' \
    "$root/lifecycle/base.response.json")
jq -e --arg run "$run_id" --arg tool_id "$base_tool_id" '
  .messages as $messages
  | any($messages[]; .role == "tool" and .tool_call_id == $tool_id
      and .content == ("probe accepted for " + $run))
' "$root/lifecycle/seed.request.json" >/dev/null || fail raw-tool-result-continuation
seed_cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens // 0' \
    "$root/lifecycle/seed.response.json")
[[ "$(jq -er '.choices[0].message.content // empty' \
      "$root/lifecycle/seed.response.json")" == CACHE_SEED_READY \
    && "$(jq -er '.choices[0].finish_reason' \
      "$root/lifecycle/seed.response.json")" == stop \
    && "$(jq -er '.usage.completion_tokens_details.reasoning_tokens // 0' \
      "$root/lifecycle/seed.response.json")" -gt 16 \
    && "$seed_cached" -ge "$minimum_cached" ]] || fail raw-seed-continuation
sed -n 's/^data: //p' "$root/lifecycle/active.stream.sse" \
  | jq -e -s 'any(.[]; type == "object" and (
      ((.choices[0].delta.content // "") | length) > 0
      or ((.choices[0].delta.reasoning_content // "") | length) > 0
      or ((.choices[0].delta.tool_calls // []) | length) > 0))' >/dev/null \
  || fail raw-semantic-sse
! rg -q '^data: \[DONE\]$' "$root/lifecycle/active.stream.sse" \
    || fail raw-cancelled-sse-terminal
sibling_cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens // 0' \
    "$root/lifecycle/sibling.response.json")
sibling_content=$(jq -er '.choices[0].message.content // empty' \
    "$root/lifecycle/sibling.response.json")
[[ "$sibling_content" == ACTIVE_STREAM_STARTED* \
    && "$(jq -er '.usage.completion_tokens_details.reasoning_tokens // 0' \
      "$root/lifecycle/sibling.response.json")" -gt 16 \
    && "$sibling_cached" -ge "$minimum_cached" ]] || fail raw-cached-retry
isolation_cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens // 0' \
    "$root/lifecycle/isolation.response.json")
[[ "$(jq -er '.choices[0].message.content // empty' \
      "$root/lifecycle/isolation.response.json")" == ISOLATION_OK \
    && "$(jq -er '.choices[0].finish_reason' \
      "$root/lifecycle/isolation.response.json")" == stop \
    && -z "$(jq -r '.choices[0].message.reasoning_content // empty' \
      "$root/lifecycle/isolation.response.json")" \
    && "$(jq -er '.usage.completion_tokens_details.reasoning_tokens // 0' \
      "$root/lifecycle/isolation.response.json")" == 0 \
    && "$isolation_cached" -le 64 ]] || fail raw-isolation
! rg -F -q -e "$run_id" -e CACHE_SEED_READY -e ACTIVE_STREAM_STARTED \
    -e ACTIVE_STREAM_COMPLETE -e lifecycle_probe \
    "$root/lifecycle/isolation.response.json" || fail raw-isolation-leak
jq -e --argjson base "$base_prompt" --argjson seed "$seed_cached" \
    --argjson sibling "$sibling_cached" --argjson isolation "$isolation_cached" '
  .base_prompt_tokens == $base and .seed_cached_tokens == $seed
  and .queued_exact_retry_cached_tokens == $sibling
  and .unrelated_conversation_cached_tokens == $isolation
  and .active_stream_cancelled_without_done == true
' "$root/lifecycle/summary.json" >/dev/null || fail raw-summary-derivation
expected_arch=$architecture
[[ "$expected_arch" == qwen35 || "$expected_arch" == qwen35moe ]] \
    || fail architecture
[[ "$(jq -er --arg arch "$expected_arch" \
    '[.data[] | select(.loaded == true and .arch == $arch)] | length' \
    "$root/models.json")" == 1 ]] || fail models-row
server_command=$(<"$root/server-command.txt")
[[ " $server_command " == *" --model $model "* \
    && " $server_command " == *" --cache-dir $root/runtime-cache "* \
    && " $server_command " == *" --scheduler inflight-batched "* \
    && " $server_command " == *" --max-slots 4 "* \
    && " $server_command " != *" --kv-persist "* ]] || fail server-command
perl -ne '
  if (/resolved serving plan/) {
    $seen++; $persist=$1 if /kv_persist_enabled=(true|false)/;
    $cache=$1 if /kv_cache_budget_bytes=([0-9]+)/;
    $budget=$1 if /kv_persist_budget_bytes=([0-9]+)/;
  }
  END {exit 1 unless $seen == 1 && $persist eq "false"
    && $cache == 51539607552 && $budget == 0}
' "$root/server.stderr" || fail serve-plan
expected_mtp=false
[[ "$architecture" == qwen35 ]] && expected_mtp=true
EXPECTED_MTP="$expected_mtp" perl -ne '
  if (/Qwen35 SlotAware prefill transaction ceiling selected/) {
    $seen++; $admit=$1 if /cross_slot_admit=(true|false)/;
    $coalesce=$1 if /cross_slot_coalesce_us=([0-9]+)/;
    $policy=$1 if /speculation_policy=(Auto|Off)/;
    $mtp=$1 if /mtp_capable=(true|false)/;
  }
  END {exit 1 unless $seen == 1 && $admit eq "true" && $coalesce == 25000
    && $policy eq "Auto" && $mtp eq $ENV{EXPECTED_MTP}}
' "$root/server.stderr" || fail slotaware-policy
EXPECTED_Q5K_POLICY="$q5k_canonical_q4x4" perl -ne '
  if (/frozen Qwen GGML routing policy/) {
    $seen++;
    $q5=$1 if /dense_q5k_canonical_q4x4=(true|false)/;
  }
  END {exit 1 unless $seen == 1 && $q5 eq $ENV{EXPECTED_Q5K_POLICY}}
' "$root/server.stderr" || fail q5k-routing-policy
qwen36_reject_fatal_log "$root/server.stderr" || fail fatal-log
events_tmp="$tmp_dir/power-events.new"
qwen36_extract_new_power_events \
    "$root/caffeinate.log.power-events.baseline" \
    "$root/caffeinate.log.power-events.final" \
    "$events_tmp" || fail power-events
cmp -s "$events_tmp" "$root/caffeinate.log.power-events.new" \
    || fail power-event-receipt
[[ ! -s "$events_tmp" ]] || fail sleep-wake-event
echo "Qwen agentic lifecycle cell: VERIFIED"
