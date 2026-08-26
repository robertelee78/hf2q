#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$ROOT_DIR/scripts/qwen38_artifact_contract.sh"
# shellcheck source=scripts/macos_thermal_guard.sh
source "$ROOT_DIR/scripts/macos_thermal_guard.sh"
# shellcheck source=scripts/qwen38_matched_reference_contract.sh
source "$ROOT_DIR/scripts/qwen38_matched_reference_contract.sh"
# shellcheck source=scripts/qwen38_matched_physical_contract.sh
source "$ROOT_DIR/scripts/qwen38_matched_physical_contract.sh"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/qwen38-matched-physical-contract.XXXXXX")
cleanup() {
    case "$TMP_DIR" in
        "${TMPDIR:-/tmp}"/qwen38-matched-physical-contract.*)
            rm -rf -- "$TMP_DIR"
            ;;
        *)
            echo "refusing to remove unexpected fixture path: $TMP_DIR" >&2
            return 1
            ;;
    esac
}
trap cleanup EXIT

fail() { echo "$*" >&2; exit 1; }
expect_reject() {
    if "$@" >/dev/null 2>&1; then fail "expected rejection: $*"; fi
}

for script in qwen38_artifact_contract.sh qwen38_matched_physical_contract.sh \
  qwen38_matched_physical_abba.sh qwen38_matched_physical_matrix.sh; do
    bash -n "$ROOT_DIR/scripts/$script"
done

# SSE transcription accepts only assistant content, exactly one terminal DONE,
# and one choice per semantic event. Tool/reasoning/refusal payloads and events
# after DONE are proof failures, even when the expected text is present.
started=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
  -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
printf '%s\n' \
  'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}' \
  'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}' \
  'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}' \
  'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":1}}' \
  'data: [DONE]' \
  | matched_physical_parse_sse_stream "$started" "$TMP_DIR/stream.sse" \
      "$TMP_DIR/stream.json" hello
jq -e '.content == "hello" and .first_semantic_ms >= 0 and .done_count == 1' \
  "$TMP_DIR/stream.json" >/dev/null
jq -n '{choices:[{message:{role:"assistant",content:"hello",refusal:null},
  finish_reason:"stop"}],usage:{prompt_tokens:7,completion_tokens:1}}' \
  >"$TMP_DIR/scalar.json"
matched_physical_validate_repeat_scalar "$TMP_DIR/stream.json" \
  "$TMP_DIR/scalar.json"
jq '.choices[0].message.content="wrong"' "$TMP_DIR/scalar.json" \
  >"$TMP_DIR/scalar-wrong.json"
expect_reject matched_physical_validate_repeat_scalar "$TMP_DIR/stream.json" \
  "$TMP_DIR/scalar-wrong.json"
jq '.choices[0].message.reasoning_content="hidden"' "$TMP_DIR/scalar.json" \
  >"$TMP_DIR/scalar-reasoning.json"
expect_reject matched_physical_validate_repeat_scalar "$TMP_DIR/stream.json" \
  "$TMP_DIR/scalar-reasoning.json"

# Canonical repeat work comes from the artifact tokenizer with special-token
# insertion disabled. It is explicitly independent of raw API usage and SSE
# frame count.
printf '%s\n' '#!/bin/sh' 'printf "%s\\n" "TOKENIZE_DEBUG_IDS: 4 9 16"' \
  >"$TMP_DIR/fake-hf2q"
chmod +x "$TMP_DIR/fake-hf2q"
printf '%s' fixture >"$TMP_DIR/model.gguf"
printf '%s' 'repeat fixture' >"$TMP_DIR/repeat.txt"
fake_sha=$(shasum -a 256 "$TMP_DIR/fake-hf2q" | awk '{print $1}')
model_sha=$(shasum -a 256 "$TMP_DIR/model.gguf" | awk '{print $1}')
matched_physical_record_semantic_repeat_tokens "$TMP_DIR/fake-hf2q" \
  "$TMP_DIR/model.gguf" "$model_sha" '1:2:3:4:5' "$fake_sha" \
  "$(printf 'a%.0s' {1..40})" "$TMP_DIR/repeat.txt" "$TMP_DIR/semantic.json"
matched_physical_validate_semantic_repeat_tokens "$TMP_DIR/semantic.json" \
  "$TMP_DIR/model.gguf" "$model_sha" '1:2:3:4:5' "$fake_sha" \
  "$(printf 'a%.0s' {1..40})" "$TMP_DIR/repeat.txt"
jq -e '.method == "hf2q-gguf-no-special-tokens-v1"
  and .semantic_completion_tokens == 3' "$TMP_DIR/semantic.json" >/dev/null
jq '.expected.bytes += 1' "$TMP_DIR/semantic.json" >"$TMP_DIR/semantic-bad.json"
expect_reject matched_physical_validate_semantic_repeat_tokens \
  "$TMP_DIR/semantic-bad.json" "$TMP_DIR/model.gguf" "$model_sha" '1:2:3:4:5' \
  "$fake_sha" "$(printf 'a%.0s' {1..40})" "$TMP_DIR/repeat.txt"

expect_bad_sse() {
    local name=$1 payload=$2
    if printf '%s\n' "$payload" 'data: [DONE]' \
      | matched_physical_parse_sse_stream "$started" "$TMP_DIR/$name.sse" \
          "$TMP_DIR/$name.json" hello >/dev/null 2>&1; then
        fail "invalid SSE passed: $name"
    fi
}
expect_bad_sse tool-call \
  'data: {"choices":[{"delta":{"content":"hello","tool_calls":[{"id":"x"}]}}]}'
expect_bad_sse reasoning \
  'data: {"choices":[{"delta":{"content":"hello","reasoning_content":"secret"}}]}'
expect_bad_sse refusal \
  'data: {"choices":[{"delta":{"content":"hello","refusal":"no"}}]}'
expect_bad_sse multiple-choices \
  'data: {"choices":[{"delta":{"content":"hello"}},{"delta":{}}]}'
if printf '%s\n' \
  'data: {"choices":[{"delta":{"content":"hello"}}]}' \
  'data: [DONE]' \
  'data: {"choices":[{"delta":{"content":"later"}}]}' \
  | matched_physical_parse_sse_stream "$started" "$TMP_DIR/post-done.sse" \
      "$TMP_DIR/post-done.json" hello >/dev/null 2>&1; then
    fail 'SSE data after DONE passed'
fi

jq -n '[{started_at:10.00,ended_at:10.20},
  {started_at:10.04,ended_at:10.18}]' >"$TMP_DIR/clients.json"
matched_physical_validate_launch_skew "$TMP_DIR/clients.json" 0.05
matched_physical_validate_client_overlap "$TMP_DIR/clients.json"
jq '.[1].started_at=10.06' "$TMP_DIR/clients.json" >"$TMP_DIR/skew-bad.json"
expect_reject matched_physical_validate_launch_skew "$TMP_DIR/skew-bad.json" 0.05
jq '.[1] = {started_at:10.21,ended_at:10.30}' "$TMP_DIR/clients.json" \
  >"$TMP_DIR/no-overlap.json"
expect_reject matched_physical_validate_client_overlap "$TMP_DIR/no-overlap.json"

printf '1.0 1\n1.1 2\n' >"$TMP_DIR/processing.tsv"
matched_physical_validate_processing_peak 2 "$TMP_DIR/processing.tsv"
printf '1.0 1\n1.1 3\n' >"$TMP_DIR/processing-bad.tsv"
expect_reject matched_physical_validate_processing_peak 2 \
  "$TMP_DIR/processing-bad.tsv"

# A monitor error must be reported only after the owned server has been reaped.
sleep 300 & owned_server=$!
(exit 17) & failed_monitor=$!
MATCHED_PHYSICAL_SERVER_INT_GRACE_SECONDS=0 \
expect_reject matched_physical_stop_owned_server "$owned_server" 61999 \
  "$failed_monitor" "$TMP_DIR/monitor.stop"
if kill -0 "$owned_server" 2>/dev/null; then fail 'server leaked after monitor failure'; fi
sleep 300 & owned_child=$!
matched_physical_terminate_owned_child "$owned_child"
if kill -0 "$owned_child" 2>/dev/null; then fail 'matrix child cleanup leaked'; fi

mkdir -p "$TMP_DIR/child-seal"
printf '%s\n' '{"verdict":"pass"}' >"$TMP_DIR/child-seal/summary.json"
printf '%s\n' payload >"$TMP_DIR/child-seal/payload.txt"
printf '%s  payload.txt\n' \
  "$(shasum -a 256 "$TMP_DIR/child-seal/payload.txt" | awk '{print $1}')" \
  >"$TMP_DIR/child-seal/evidence.sha256"
printf '%s  summary.json\n%s  evidence.sha256\n' \
  "$(shasum -a 256 "$TMP_DIR/child-seal/summary.json" | awk '{print $1}')" \
  "$(shasum -a 256 "$TMP_DIR/child-seal/evidence.sha256" | awk '{print $1}')" \
  >"$TMP_DIR/child-seal/result.sha256"
matched_physical_require_child_seal "$TMP_DIR/child-seal"
printf '%s  payload.txt\n%s  payload.txt\n' \
  "$(shasum -a 256 "$TMP_DIR/child-seal/payload.txt" | awk '{print $1}')" \
  "$(shasum -a 256 "$TMP_DIR/child-seal/payload.txt" | awk '{print $1}')" \
  >"$TMP_DIR/child-seal/result.sha256"
expect_reject matched_physical_require_child_seal "$TMP_DIR/child-seal"

write_hf2q_metrics() {
    local path=$1 proposals=$2 drafted=$3 accepted=$4 disabled=$5 round=$6 ordinary=$7
    : >"$path"
    for proposer in history_lookup mtp; do
        {
            printf '%s{proposer="%s"} %s\n' \
              hf2q_qwen_speculation_proposals_total "$proposer" "$proposals"
            printf '%s{proposer="%s"} %s\n' \
              hf2q_qwen_speculation_drafted_tokens_total "$proposer" "$drafted"
            printf '%s{proposer="%s"} %s\n' \
              hf2q_qwen_speculation_accepted_tokens_total "$proposer" "$accepted"
            printf '%s{proposer="%s"} %s\n' \
              hf2q_qwen_speculation_cost_disabled_total "$proposer" "$disabled"
            printf '%s{proposer="%s"} %s\n' \
              hf2q_qwen_speculation_round_seconds_total "$proposer" "$round"
            printf '%s{proposer="%s"} %s\n' \
              hf2q_qwen_speculation_equivalent_ordinary_seconds_total "$proposer" \
              "$ordinary"
        } >>"$path"
    done
}
write_reference_metrics() {
    local path=$1 proposals=$2 drafted=$3 accepted=$4
    printf '%s\n' \
      "llamacpp:spec_decode_num_drafts_total $proposals" \
      "llamacpp:spec_decode_num_draft_tokens_total $drafted" \
      "llamacpp:spec_decode_num_accepted_tokens_total $accepted" >"$path"
}
write_hf2q_metrics "$TMP_DIR/h-before" 0 0 0 0 0 0
write_hf2q_metrics "$TMP_DIR/h-accepted" 2 4 3 0 1.0 0.5
matched_physical_validate_wave_speculation hf2q 1 code "$TMP_DIR/h-before" \
  "$TMP_DIR/h-accepted" "$TMP_DIR/h-accepted.json"
jq -e '.group == "code"
  and .policy == "adaptive-history-then-mtp-cost-gated"
  and .proof_mode == "accepted-proposals" and .accepted_tokens == 6' \
  "$TMP_DIR/h-accepted.json" >/dev/null
write_hf2q_metrics "$TMP_DIR/h-disabled" 2 4 0 1 1.0 0.5
matched_physical_validate_wave_speculation hf2q 4 repeat "$TMP_DIR/h-before" \
  "$TMP_DIR/h-disabled" "$TMP_DIR/h-disabled.json"
jq -e '.proof_mode == "measured-cost-disabled"
  and .disable_reason == "measured_cost_unprofitable"' \
  "$TMP_DIR/h-disabled.json" >/dev/null
expect_reject matched_physical_validate_wave_speculation hf2q 1 code \
  "$TMP_DIR/h-before" "$TMP_DIR/h-before" "$TMP_DIR/no-spec.json"
write_hf2q_metrics "$TMP_DIR/h-fake-disable" 2 4 0 1 0 0
expect_reject matched_physical_validate_wave_speculation hf2q 1 code \
  "$TMP_DIR/h-before" "$TMP_DIR/h-fake-disable" "$TMP_DIR/fake-disable.json"
write_reference_metrics "$TMP_DIR/r-before" 0 0 0
write_reference_metrics "$TMP_DIR/r-after" 2 4 3
matched_physical_validate_wave_speculation reference 2 code "$TMP_DIR/r-before" \
  "$TMP_DIR/r-after" "$TMP_DIR/r.json"
jq -e '.policy == "fixed-k3-mtp" and .group == "code"' "$TMP_DIR/r.json" \
  >/dev/null
write_reference_metrics "$TMP_DIR/r-inactive" 2 4 0
expect_reject matched_physical_validate_wave_speculation reference 2 code \
  "$TMP_DIR/r-before" "$TMP_DIR/r-inactive" "$TMP_DIR/r-inactive.json"
expect_reject matched_physical_validate_wave_speculation reference 2 warmup \
  "$TMP_DIR/r-before" "$TMP_DIR/r-after" "$TMP_DIR/r-warmup.json"

write_rows() {
    local width=$1 output=$2 group trial engine completion semantic client_wall wave_wall ttft
    local clients api_total comparison_work_units comparison_rate api_rate unit
    : >"$output"
    for group in code repeat; do
        for trial in 1 2 3 4; do
            case "$trial" in 1|4) engine=hf2q ;; *) engine=reference ;; esac
            if [[ "$group" == code && "$engine" == hf2q ]]; then
                completion=100; client_wall=1.8; wave_wall=2; ttft=null
            elif [[ "$group" == code ]]; then
                completion=100; client_wall=3.8; wave_wall=4; ttft=null
            elif [[ "$engine" == hf2q ]]; then
                completion=100; client_wall=0.8; wave_wall=1; ttft=10
            else
                completion=101; client_wall=1.8; wave_wall=2; ttft=12
            fi
            semantic=100
            clients=$(jq -nc --argjson width "$width" --argjson wall "$client_wall" \
              --argjson ttft "$ttft" --argjson completion "$completion" \
              --argjson semantic "$semantic" --arg group "$group" \
              '[range(1;$width+1) | {lane:.,started_at:10,
                ended_at:(10+$wall),wall_seconds:$wall,prompt_tokens:50,
                completion_tokens:$completion,first_semantic_ms:$ttft,
                scalar_parity:true}
                + (if $group == "repeat" then {semantic_completion_tokens:$semantic,
                  semantic_tokenization_sha256:("d" * 64)} else {} end)]')
            api_total=$((width * completion))
            if [[ "$group" == repeat ]]; then
                comparison_work_units=$((width * semantic))
                unit='canonical-semantic-output-token'
            else
                comparison_work_units=$width
                unit='evaluator-valid-code-request'
            fi
            comparison_rate=$(awk -v total="$comparison_work_units" \
              -v wall="$wave_wall" 'BEGIN {printf "%.9f", total/wall}')
            api_rate=$(awk -v total="$api_total" -v wall="$wave_wall" \
              'BEGIN {printf "%.9f", total/wall}')
            jq -nc --arg engine "$engine" --arg group "$group" \
              --argjson trial "$trial" --argjson width "$width" \
              --argjson wave_start 9.9 --argjson wave_end "$(awk -v wall="$wave_wall" \
                'BEGIN {print 9.9+wall}')" \
              --argjson wave "$wave_wall" --arg unit "$unit" \
              --argjson api_total "$api_total" \
              --argjson comparison_work_units "$comparison_work_units" \
              --argjson comparison_rate "$comparison_rate" \
              --argjson api_rate "$api_rate" --argjson clients "$clients" '{
                schema:1,width:$width,engine:$engine,trial:$trial,group:$group,
                quality_pass:true,api_concurrency_proven:true,
                wave_started_at:$wave_start,wave_ended_at:$wave_end,
                wave_wall_seconds:$wave,comparison_unit:$unit,
                total_completion_tokens:$api_total,
                total_semantic_completion_tokens:(if $group == "repeat"
                  then $comparison_work_units else null end),
                comparison_work_units:$comparison_work_units,
                comparison_units_per_second:$comparison_rate,
                diagnostics:{api_completion_tokens_per_second:$api_rate},
                clients:$clients}' >>"$output"
        done
    done
}

write_rows 2 "$TMP_DIR/rows.jsonl"
code=$(matched_physical_group_result_json "$TMP_DIR/rows.jsonl" 2 code 5 10)
repeat=$(matched_physical_group_result_json "$TMP_DIR/rows.jsonl" 2 repeat 5 10)
jq -e '.measurement_consistent and .api_concurrency_pass
  and .token_accounting.pass and .stability.stable
  and .stability.observed_band_dominance
  and .hf2q_over_reference_comparison_rate == 2' <<<"$code" >/dev/null
jq -e '.measurement_consistent and .api_concurrency_pass
  and .token_accounting.pass and .stability.stable
  and .stability.observed_band_dominance
  and .reference_over_hf2q_p95_wall >= 1
  and .semantic_ttft.required and .semantic_ttft.stable
  and .semantic_ttft.observed_band_dominance' <<<"$repeat" >/dev/null

jq -c 'if .group=="repeat" and .engine=="reference" and .trial==2
  then .clients[0].prompt_tokens=51 else . end' "$TMP_DIR/rows.jsonl" \
  >"$TMP_DIR/prompt-drift.jsonl"
bad=$(matched_physical_group_result_json "$TMP_DIR/prompt-drift.jsonl" 2 repeat 5 10)
jq -e '.token_accounting.pass == false and .stability.stable == false' \
  <<<"$bad" >/dev/null
jq -c 'if .group=="repeat" and .engine=="reference" and .trial==2
  then .clients[0].completion_tokens=102 | .total_completion_tokens=203
  else . end' "$TMP_DIR/rows.jsonl" \
  >"$TMP_DIR/completion-drift.jsonl"
bad=$(matched_physical_group_result_json "$TMP_DIR/completion-drift.jsonl" 2 repeat 5 10)
jq -e '.token_accounting.pass == false and .stability.stable == false' \
  <<<"$bad" >/dev/null
# Cross-engine API counts are intentionally not a repeat or code-work
# comparator. The fixed repeat has a canonical semantic count; code uses one
# evaluator-valid response as its common work unit. In both groups, raw counts
# remain diagnostic and must nevertheless be stable inside each engine's ABBA
# pair.
jq -c 'if .group=="code" and .engine=="reference"
  then .clients[0].completion_tokens=101 | .clients[1].completion_tokens=101
    | .total_completion_tokens=202
    | .diagnostics.api_completion_tokens_per_second=50.5
  else . end' "$TMP_DIR/rows.jsonl" >"$TMP_DIR/code-cross-engine-api-drift.jsonl"
code=$(matched_physical_group_result_json "$TMP_DIR/code-cross-engine-api-drift.jsonl" 2 code 5 10)
jq -e '.token_accounting.pass == true and .stability.stable == true
  and .token_accounting.cross_engine_completion_equality_required == false
  and .hf2q_over_reference_comparison_rate == 2
  and .diagnostics.reference_median_api_completion_tokens_per_second == 50.5' \
  <<<"$code" >/dev/null
jq -c 'if .group=="repeat" and .engine=="reference" and .trial==2
  then .clients[0].semantic_completion_tokens=99
    | .total_semantic_completion_tokens=199
    | .comparison_work_units=199
    | .comparison_units_per_second=99.5 else . end' "$TMP_DIR/rows.jsonl" \
  >"$TMP_DIR/semantic-drift.jsonl"
bad=$(matched_physical_group_result_json "$TMP_DIR/semantic-drift.jsonl" 2 repeat 5 10)
jq -e '.token_accounting.pass == false and .stability.stable == false' \
  <<<"$bad" >/dev/null
jq -c 'if .group=="repeat" and .engine=="reference" and .trial==2
  then .clients[0].semantic_tokenization_sha256=("e" * 64) else . end' \
  "$TMP_DIR/rows.jsonl" >"$TMP_DIR/semantic-binding-drift.jsonl"
bad=$(matched_physical_group_result_json "$TMP_DIR/semantic-binding-drift.jsonl" 2 repeat 5 10)
jq -e '.token_accounting.pass == false and .stability.stable == false' \
  <<<"$bad" >/dev/null
jq -c 'if .group=="code" and .engine=="hf2q" and .trial==1
  then .comparison_units_per_second=999 else . end' "$TMP_DIR/rows.jsonl" \
  >"$TMP_DIR/bad-denominator.jsonl"
bad=$(matched_physical_group_result_json "$TMP_DIR/bad-denominator.jsonl" 2 code 5 10)
jq -e '.measurement_consistent == false and .stability.stable == false' \
  <<<"$bad" >/dev/null
jq -c 'if .group=="code" and .engine=="hf2q" and .trial==1
  then .wave_wall_seconds=2.1 | .comparison_units_per_second=(2/2.1)
    | .diagnostics.api_completion_tokens_per_second=(200/2.1)
  else . end' "$TMP_DIR/rows.jsonl" >"$TMP_DIR/bad-boundary-wall.jsonl"
bad=$(matched_physical_group_result_json "$TMP_DIR/bad-boundary-wall.jsonl" 2 code 5 10)
jq -e '.measurement_consistent == false and .stability.stable == false' \
  <<<"$bad" >/dev/null
jq -c 'if .group=="repeat" and .engine=="hf2q" and .trial==1
  then .clients[0].wall_seconds=0.7 else . end' "$TMP_DIR/rows.jsonl" \
  >"$TMP_DIR/bad-client-wall.jsonl"
bad=$(matched_physical_group_result_json "$TMP_DIR/bad-client-wall.jsonl" 2 repeat 5 10)
jq -e '.measurement_consistent == false and .stability.stable == false' \
  <<<"$bad" >/dev/null
jq -c 'if .group=="code" and .engine=="hf2q" and .trial==1
  then .api_concurrency_proven=false else . end' "$TMP_DIR/rows.jsonl" \
  >"$TMP_DIR/no-concurrency.jsonl"
bad=$(matched_physical_group_result_json "$TMP_DIR/no-concurrency.jsonl" 2 code 5 10)
jq -e '.api_concurrency_pass == false and .stability.stable == false' \
  <<<"$bad" >/dev/null

speculation=$(jq -nc --slurpfile h1 "$TMP_DIR/h-accepted.json" \
  --slurpfile r2 "$TMP_DIR/r.json" --slurpfile h4 "$TMP_DIR/h-disabled.json" '
  {hf2q_policy:"adaptive-history-then-mtp-cost-gated",
    reference_policy:"fixed-k3-mtp",
    waves:[$h1[0],($h1[0] | .group="repeat"),
      $r2[0],($r2[0] | .group="repeat"),
      ($r2[0] | .trial=3),($r2[0] | .trial=3 | .group="repeat"),
      ($h4[0] | .group="code"),$h4[0]]}')
results='[]'
for width in 1 2 4 8 16; do
    write_rows "$width" "$TMP_DIR/rows-$width.jsonl"
    code=$(matched_physical_group_result_json "$TMP_DIR/rows-$width.jsonl" \
      "$width" code 5 10)
    repeat=$(matched_physical_group_result_json "$TMP_DIR/rows-$width.jsonl" \
      "$width" repeat 5 10)
    clients=$(jq -nc --argjson width "$width" \
      '[range(1;$width+1) | {lane:.,scalar_parity:true}]')
    cell=$(jq -nc --argjson width "$width" --argjson clients "$clients" \
      --argjson code "$code" --argjson repeat "$repeat" \
      --argjson speculation "$speculation" '{
        schema:2,verdict:"pass",width:$width,samples:{hf2q:2,reference:2},
        scalar_replay:{hf2q:true,reference:true},
        hf2q_effective_routing_policy:{dense_decode_mvn:1,
          dense_decode_mv_ext:0,dense_q5k_canonical_q4x4:1},
        physical_proof:{width:$width,mode:"ordinary-target-speculation-off",
          seal_validated:true,scheduler_max_width:$width,
          target_body_max_width:$width,target_head_max_width:$width,
          command_buffer_submissions_delta:1,clients:$clients},
        speculation:$speculation,acceptance:{minimum_hf2q_ratio:1},
        code:$code,repeat:$repeat}')
    results=$(jq -nc --argjson prior "$results" --argjson cell "$cell" \
      '$prior + [$cell]')
done
jq -n --argjson results "$results" '{schema:2,verdict:"pass",
  gate:"qwen38-matched-physical-abba",
  harness:{commit:("0"*40),source_binding:"clean exact harness worktree"},
  hf2q:{commit:("a"*40),binary_sha256:("b"*64),
    effective_routing_policy:{dense_decode_mvn:1,dense_decode_mv_ext:0,
      dense_q5k_canonical_q4x4:1}},
  reference:{commit:("c"*40),binary_sha256:("d"*64),
    runtime_manifest_sha256:("e"*64),
    expected_runtime_manifest_sha256:("e"*64),
    pin_policy:"observed-current-then-frozen",frozen_for_run:true,
    pin_file_sha256:("f"*64)},
  physical_matrix_sha256:("1"*64),
  workload:{widths:[1,2,4,8,16],
    trial_order:["hf2q","reference","reference","hf2q"],
    speculation:{hf2q:"adaptive-history-then-mtp-cost-gated",
      reference:"fixed-k3-mtp"},
    cache_settings:{
      hf2q:{format:"tq-kv",budget_bytes:51539607552,
        context_tokens_per_slot:262144},
      reference:{k_format:"q8_0",v_format:"q8_0",
        context_tokens_total:262144}},
    scalar_replay_per_lane:true,
    repeat_semantic_tokenization:{receipt_sha256:("d"*64),completion_tokens:100,
      unit:"canonical-semantic-output-token"},
    reference_parallelism_matches_width:true},
  acceptance:{minimum_hf2q_ratio:1,maximum_launch_skew_seconds:0.1},
  host_contention:{policy:"process-group-cpu-v2",
    maximum_foreign_cpu_percent:100,
    owner_scope:"release-gate-process-group",owner_pgid:100,continuous:true},
  evidence:{reference_runtime_manifest_sha256:("e"*64),
    expected_reference_runtime_manifest_sha256:("e"*64),
    reference_pin_file_sha256:("f"*64)},results:$results}' \
  >"$TMP_DIR/summary.json"
matched_physical_validate_inner_summary "$TMP_DIR/summary.json"

# A coherent rehash cannot convert contended raw telemetry into performance
# authority. Build all 20 expected trials, prove the reopened validator passes,
# then mutate and fully reseal one middle measurement log.
reopened_child="$TMP_DIR/reopened-child"
mkdir -p "$reopened_child"
cp "$TMP_DIR/summary.json" "$reopened_child/summary.json"
: >"$reopened_child/contention-preflight.tsv"
for sample in $(seq 1 22); do
    printf '%s\tquiet\tpreflight\t100\t99.9\t-\n' "$sample" \
      >>"$reopened_child/contention-preflight.tsv"
done
for width in 1 2 4 8 16; do
    trial=0
    for engine in hf2q reference reference hf2q; do
        trial=$((trial + 1))
        trial_dir="$reopened_child/widths/width-$width/trials/trial-$trial-$engine"
        mkdir -p "$trial_dir/code/code-validation" "$trial_dir/repeat/responses"
        : >"$trial_dir/thermal-settle.tsv"
        : >"$trial_dir/host-settle.tsv"
        : >"$trial_dir/contention-settle.tsv"
        for sampled_at in $(seq 1000 5 1120); do
            printf '%s\tnominal\tloaded-idle\n' "$sampled_at" \
              >>"$trial_dir/thermal-settle.tsv"
            printf '%s\tac\tquiet\tautomatic\t0\tloaded-idle\n' "$sampled_at" \
              >>"$trial_dir/host-settle.tsv"
            printf '%s\tquiet\tloaded-idle\t100\t99.9\t-\n' "$sampled_at" \
              >>"$trial_dir/contention-settle.tsv"
        done
        : >"$trial_dir/thermal-measurement.tsv"
        : >"$trial_dir/host-measurement.tsv"
        : >"$trial_dir/contention-measurement.tsv"
        for phase_row in '2000 measurement-start' '2002 measurement' \
          '2004 measurement-end'; do
            sampled_at=${phase_row%% *}
            phase=${phase_row#* }
            printf '%s\tnominal\t%s\n' "$sampled_at" "$phase" \
              >>"$trial_dir/thermal-measurement.tsv"
            printf '%s\tac\tquiet\tautomatic\t0\t%s\n' "$sampled_at" "$phase" \
              >>"$trial_dir/host-measurement.tsv"
            printf '%s\tquiet\t%s\t100\t99.9\t-\n' "$sampled_at" "$phase" \
              >>"$trial_dir/contention-measurement.tsv"
        done
    done
done
seal_reopened_child() {
    local root=$1 path
    : >"$root/evidence.sha256"
    while IFS= read -r path; do
        printf '%s  %s\n' \
          "$(shasum -a 256 "$root/$path" | awk '{print $1}')" "$path" \
          >>"$root/evidence.sha256"
    done < <(cd "$root" && find . -type f \
      ! -name summary.json ! -name evidence.sha256 ! -name result.sha256 \
      -print | sed 's#^./##' | sort)
    printf '%s  summary.json\n%s  evidence.sha256\n' \
      "$(shasum -a 256 "$root/summary.json" | awk '{print $1}')" \
      "$(shasum -a 256 "$root/evidence.sha256" | awk '{print $1}')" \
      >"$root/result.sha256"
}
seal_reopened_child "$reopened_child"
matched_physical_validate_reopened_child "$reopened_child"
ln -s "$reopened_child" "$TMP_DIR/reopened-child-symlink"
expect_reject matched_physical_validate_reopened_child \
  "$TMP_DIR/reopened-child-symlink"
mutated_contention="$reopened_child/widths/width-8/trials/trial-3-reference/contention-measurement.tsv"
awk -F '\t' 'BEGIN { OFS="\t" }
  NR == 2 { $2="contended"; $5="100.0"; $6="999:888:browser" }
  { print }' "$mutated_contention" >"$mutated_contention.tmp"
mv "$mutated_contention.tmp" "$mutated_contention"
seal_reopened_child "$reopened_child"
matched_physical_require_child_seal "$reopened_child"
expect_reject matched_physical_validate_reopened_child "$reopened_child"
awk -F '\t' 'BEGIN { OFS="\t" }
  { $2="quiet"; $4="100"; $5="99.9"; $6="-"; print }
' "$mutated_contention" >"$mutated_contention.tmp"
mv "$mutated_contention.tmp" "$mutated_contention"
seal_reopened_child "$reopened_child"
omitted_path='widths/width-8/trials/trial-3-reference/contention-measurement.tsv'
awk -v omitted="$omitted_path" '$2 != omitted { print }' \
  "$reopened_child/evidence.sha256" >"$reopened_child/evidence.sha256.tmp"
mv "$reopened_child/evidence.sha256.tmp" "$reopened_child/evidence.sha256"
printf '%s  summary.json\n%s  evidence.sha256\n' \
  "$(shasum -a 256 "$reopened_child/summary.json" | awk '{print $1}')" \
  "$(shasum -a 256 "$reopened_child/evidence.sha256" | awk '{print $1}')" \
  >"$reopened_child/result.sha256"
matched_physical_require_child_seal "$reopened_child"
expect_reject matched_physical_validate_reopened_child "$reopened_child"
awk -F '\t' 'BEGIN { OFS="\t" }
  { $2="quiet"; $4="101"; $5="99.9"; $6="-"; print }
' "$mutated_contention" >"$mutated_contention.tmp"
mv "$mutated_contention.tmp" "$mutated_contention"
host_contention_validate_measurement_log "$mutated_contention" 5
seal_reopened_child "$reopened_child"
matched_physical_require_child_seal "$reopened_child"
expect_reject matched_physical_validate_reopened_child "$reopened_child"

expected_runtime_closure=$(printf 'e%.0s' {1..64})
matched_physical_validate_expected_reference_closure \
  "$TMP_DIR/summary.json" "$expected_runtime_closure"
expect_reject matched_physical_validate_expected_reference_closure \
  "$TMP_DIR/summary.json" not-a-sha256
for mutation in old-schema empty-clients lane-zero wrong-mode unsealed no-spec \
  fake-cost-disable width-mismatch token-proof concurrency-proof weak-threshold \
  live-tip-policy mutable-pin runtime-closure-drift semantic-tokenization \
  semantic-tokenization-binding wrong-hf2q-policy wrong-reference-policy \
  wrong-kv-budget wrong-reference-cache wrong-context-scope wrong-launch-skew \
  wrong-effective-q5k wrong-width-effective-q5k stale-contention-policy \
  weak-contention-threshold invalid-contention-owner \
  noncontinuous-contention; do
    case "$mutation" in
      old-schema) filter='.schema=1' ;;
      empty-clients) filter='.results[2].physical_proof.clients=[]' ;;
      lane-zero) filter='.results[2].physical_proof.clients[0].lane=0' ;;
      wrong-mode) filter='.results[2].physical_proof.mode="speculative"' ;;
      unsealed) filter='.results[2].physical_proof.seal_validated=false' ;;
      no-spec) filter='.results[2].speculation.waves[0].proposals=0' ;;
      fake-cost-disable) filter='.results[2].speculation.waves[7].measured_round_seconds=0' ;;
      width-mismatch) filter='.results[2].code.width=99' ;;
      token-proof) filter='.results[2].repeat.token_accounting.pass=false' ;;
      concurrency-proof) filter='.results[2].code.api_concurrency_pass=false' ;;
      weak-threshold) filter='.results[2].acceptance.minimum_hf2q_ratio=0.99' ;;
      live-tip-policy) filter='.reference.pin_policy="live-tip"' ;;
      mutable-pin) filter='.reference.frozen_for_run=false' ;;
      runtime-closure-drift) filter='.reference.runtime_manifest_sha256=("9"*64)' ;;
      semantic-tokenization) filter='.workload.repeat_semantic_tokenization.completion_tokens=0' ;;
      semantic-tokenization-binding) filter='.workload.repeat_semantic_tokenization.receipt_sha256=("e"*64)' ;;
      wrong-hf2q-policy) filter='.workload.speculation.hf2q="fixed-k3-mtp"' ;;
      wrong-reference-policy) filter='.results[2].speculation.reference_policy="shipping-auto"' ;;
      wrong-kv-budget) filter='.workload.cache_settings.hf2q.budget_bytes += 1' ;;
      wrong-reference-cache) filter='.workload.cache_settings.reference.k_format="tq-kv"' ;;
      wrong-context-scope) filter='.workload.cache_settings.reference.context_tokens_total=16384' ;;
      wrong-launch-skew) filter='.acceptance.maximum_launch_skew_seconds=0.2' ;;
      wrong-effective-q5k) filter='.hf2q.effective_routing_policy.dense_q5k_canonical_q4x4=0' ;;
      wrong-width-effective-q5k) filter='.results[2].hf2q_effective_routing_policy.dense_q5k_canonical_q4x4=0' ;;
      stale-contention-policy) filter='.host_contention.policy="process-group-v1"' ;;
      weak-contention-threshold) filter='.host_contention.maximum_foreign_cpu_percent=101' ;;
      invalid-contention-owner) filter='.host_contention.owner_pgid=0' ;;
      noncontinuous-contention) filter='.host_contention.continuous=false' ;;
    esac
    jq "$filter" "$TMP_DIR/summary.json" >"$TMP_DIR/$mutation.json"
    expect_reject matched_physical_validate_inner_summary "$TMP_DIR/$mutation.json"
done

# Internal equality is necessary but insufficient: a coherently rewritten
# child must still fail the externally frozen runtime-closure identity.
jq '.reference.runtime_manifest_sha256=("9"*64)
  | .reference.expected_runtime_manifest_sha256=("9"*64)
  | .evidence.reference_runtime_manifest_sha256=("9"*64)
  | .evidence.expected_reference_runtime_manifest_sha256=("9"*64)' \
  "$TMP_DIR/summary.json" >"$TMP_DIR/coherent-wrong-closure.json"
matched_physical_validate_inner_summary "$TMP_DIR/coherent-wrong-closure.json"
expect_reject matched_physical_validate_expected_reference_closure \
  "$TMP_DIR/coherent-wrong-closure.json" "$expected_runtime_closure"

jq -n --slurpfile child "$TMP_DIR/summary.json" \
  --arg expected "$expected_runtime_closure" '
  ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"] as $formats
  | {schema:2,verdict:"pass",
      gate:"qwen38-matched-physical-artifact-matrix",
      harness:{commit:("0"*40),source_binding:"clean exact harness worktree"},
      reference_runtime_manifest_sha256:$expected,
      hf2q_effective_routing_policy:{dense_decode_mvn:1,
        dense_decode_mv_ext:0,dense_q5k_canonical_q4x4:1},
      results:($formats | map(. as $format
        | $child[0] | .model={format:$format}))}' \
  >"$TMP_DIR/matrix-summary.json"
matched_physical_validate_matrix_reference_cohort \
  "$TMP_DIR/matrix-summary.json" "$expected_runtime_closure"
jq '.results[2].reference.runtime_manifest_sha256=("9"*64)
  | .results[2].reference.expected_runtime_manifest_sha256=("9"*64)
  | .results[2].evidence.reference_runtime_manifest_sha256=("9"*64)
  | .results[2].evidence.expected_reference_runtime_manifest_sha256=("9"*64)' \
  "$TMP_DIR/matrix-summary.json" >"$TMP_DIR/matrix-mixed-closure.json"
expect_reject matched_physical_validate_matrix_reference_cohort \
  "$TMP_DIR/matrix-mixed-closure.json" "$expected_runtime_closure"
jq '.results[2].reference.pin_file_sha256=("8"*64)
  | .results[2].evidence.reference_pin_file_sha256=("8"*64)' \
  "$TMP_DIR/matrix-summary.json" >"$TMP_DIR/matrix-mixed-pin.json"
expect_reject matched_physical_validate_matrix_reference_cohort \
  "$TMP_DIR/matrix-mixed-pin.json" "$expected_runtime_closure"
jq '.results[2].harness.commit=("7"*40)' "$TMP_DIR/matrix-summary.json" \
  >"$TMP_DIR/matrix-mixed-harness.json"
expect_reject matched_physical_validate_matrix_reference_cohort \
  "$TMP_DIR/matrix-mixed-harness.json" "$expected_runtime_closure"
jq '.reference_runtime_manifest_sha256=("9"*64)' \
  "$TMP_DIR/matrix-summary.json" >"$TMP_DIR/matrix-wrong-expected.json"
expect_reject matched_physical_validate_matrix_reference_cohort \
  "$TMP_DIR/matrix-wrong-expected.json" "$expected_runtime_closure"

# Static order checks bind the source to the source-only contracts.
runner="$ROOT_DIR/scripts/qwen38_matched_physical_abba.sh"
unmarked_error="$TMP_DIR/unmarked-bootstrap.err"
if "$runner" >/dev/null 2>"$unmarked_error"; then
    fail 'unconfigured matched physical runner unexpectedly succeeded'
fi
grep -Fq 'HF2Q_BIN is required' "$unmarked_error" \
    || fail 'matched physical runner did not self-bootstrap before environment validation'
marked_error="$TMP_DIR/marked-bootstrap.err"
if HF2Q_MATCHED_GATE_ISOLATED=1 "$runner" \
    >/dev/null 2>"$marked_error"; then
    fail 'forced physical isolation marker unexpectedly succeeded'
fi
grep -Fq 'does not own an isolated process group' "$marked_error" \
    || fail 'forced physical isolation marker bypassed group leadership proof'
matrix_runner="$ROOT_DIR/scripts/qwen38_matched_physical_matrix.sh"
[[ "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" == 1 ]] \
  || fail 'matched physical Q5_K policy is not uniformly enabled'
rg -F 'env -i' "$runner" >/dev/null \
  || fail 'matched physical runner does not launch from a clean environment'
run_wave_line=$(rg -n '^run_wave\(\)' "$runner" | cut -d: -f1)
before_line=$(rg -n '"\$speculation_before"$' "$runner" | head -1 | cut -d: -f1)
release_line=$(rg -n '^[[:space:]]+: >"\$start_file"' "$runner" | tail -1 | cut -d: -f1)
after_line=$(rg -n '"\$speculation_after"$' "$runner" | tail -1 | cut -d: -f1)
spec_validate_line=$(rg -n 'matched_physical_validate_wave_speculation ' "$runner" \
  | cut -d: -f1)
cache_replay_line=$(rg -n 'run_cache_replacement "\$width"' "$runner" | cut -d: -f1)
next_function_line=$(rg -n '^validate_trial_code_quality\(\)' "$runner" | cut -d: -f1)
stop_line=$(rg -n '^[[:space:]]+stop_server$' "$runner" | tail -1 | cut -d: -f1)
quality_line=$(rg -n '^[[:space:]]+validate_trial_code_quality ' "$runner" \
  | tail -1 | cut -d: -f1)
((run_wave_line < before_line && before_line < release_line \
  && release_line < after_line && after_line < spec_validate_line \
  && spec_validate_line < cache_replay_line && cache_replay_line < next_function_line \
  && stop_line < quality_line))
if rg -q 'matched_physical_validate_trial_speculation|trial_dir/speculation.json' \
  "$runner"; then
    fail 'trial-wide speculation proof survived'
fi
rg -F 'QWEN38_MATCHED_HF2Q_SPECULATION_POLICY' "$runner" >/dev/null
rg -F 'QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY' "$runner" >/dev/null
rg -F 'HF2Q_DECODE_MVN="$QWEN38_PHYSICAL_DECODE_MVN"' "$runner" >/dev/null
rg -F 'HF2Q_DECODE_MV_EXT="$QWEN38_PHYSICAL_DECODE_MV_EXT"' "$runner" >/dev/null
rg -F 'HF2Q_Q5K_CANONICAL_Q4X4="$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4"' \
  "$runner" >/dev/null
rg -F 'matched_validate_qwen_frozen_routing_policy_log "$log"' \
  "$runner" >/dev/null
rg -F -- '--ctx-size "$QWEN38_MATCHED_CONTEXT_TOKENS"' "$runner" >/dev/null
rg -F -- '--cache-type-k "$QWEN38_MATCHED_REFERENCE_KV_CACHE_K"' \
  "$runner" >/dev/null
rg -F -- '--cache-type-v "$QWEN38_MATCHED_REFERENCE_KV_CACHE_V"' \
  "$runner" >/dev/null
if rg -q 'speculation:"shipping-auto"|policy:"shipping-auto"' "$runner" \
  "$ROOT_DIR/scripts/qwen38_matched_physical_contract.sh"; then
    fail 'undifferentiated speculation policy survived'
fi
rg -F 'matched_physical_validate_client_overlap' "$runner" >/dev/null
rg -F 'HOST_CONTENTION_GATE_OWNER_PID' "$runner" >/dev/null \
  || fail 'matched physical runner does not use one stable process-group owner'
rg -F 'matched_physical_validate_reopened_child "$OUT_DIR"' "$runner" >/dev/null \
  || fail 'matched physical runner does not semantically reopen its evidence'
rg -F '[[ -d "$matrix_dir/artifacts" && ! -L "$matrix_dir/artifacts" ]]' \
  "$ROOT_DIR/scripts/qwen38_matched_physical_contract.sh" >/dev/null \
  || fail 'matched physical matrix reopener follows a symlinked artifact root'
for contention_call in \
  'host_contention_sample ' \
  'host_contention_require_quiet ' \
  'host_contention_validate_settle_log ' \
  'host_contention_validate_measurement_log ' \
  'host_contention_validate_thermal_alignment '; do
    rg -F "$contention_call" "$runner" >/dev/null \
      || fail "matched physical runner omits v2 contention authority: $contention_call"
done
rg -F 'matched_record_calibration_observation "$1" "$2" "$3" "$4"' \
  "$runner" >/dev/null \
  || fail 'matched physical runner bypasses the shared calibration predicate'
rg -F '"$THERMAL_SAMPLED_AT" "$owned_server_pid"' \
  "$ROOT_DIR/scripts/qwen38_matched_reference_contract.sh" >/dev/null \
  || fail 'shared calibration predicate does not narrowly exempt its owned server PID'
if rg -Fq 'require_no_foreign_heavy_work ' "$runner"; then
    fail 'matched physical runner still uses the name-only contention predicate'
fi
rg -F 'hf2q_macos_verify_runtime_manifest "$REFERENCE_BIN"' "$runner" \
  >/dev/null
rg -F 'runtime_manifest_sha256:$reference_runtime_manifest_sha' "$runner" \
  >/dev/null
rg -F 'HF2Q_SOURCE_DIR=${HF2Q_SOURCE_DIR:?HF2Q_SOURCE_DIR is required}' \
  "$runner" >/dev/null
for source_runner in "$runner" "$matrix_runner"; do
    rg -F 'REFERENCE_RUNTIME_MANIFEST_SHA256=${REFERENCE_RUNTIME_MANIFEST_SHA256:?REFERENCE_RUNTIME_MANIFEST_SHA256 is required}' \
      "$source_runner" >/dev/null
done
rg -F '"$HF2Q_SOURCE_DIR/scripts/serve_qwen38_opencode.sh"' "$runner" \
  >/dev/null
rg -F 'HF2Q_SOURCE_DIR="$HF2Q_SOURCE_DIR"' "$matrix_runner" >/dev/null
rg -F 'REFERENCE_RUNTIME_MANIFEST_SHA256="$REFERENCE_RUNTIME_MANIFEST_SHA256"' \
  "$matrix_runner" >/dev/null
closure_line=$(rg -n 'reference runtime closure mismatch:' "$runner" \
  | head -1 | cut -d: -f1)
output_line=$(rg -n 'mkdir -p "\$OUT_DIR/requests/code"' "$runner" \
  | head -1 | cut -d: -f1)
((closure_line < output_line))
matrix_closure_line=$(rg -n 'reference runtime closure mismatch:' \
  "$matrix_runner" | head -1 | cut -d: -f1)
matrix_output_line=$(rg -n 'mkdir -p "\$OUT_DIR/artifacts"' "$matrix_runner" \
  | head -1 | cut -d: -f1)
((matrix_closure_line < matrix_output_line))
rg -F 'reference_runtime_manifest_final_sha=' "$runner" "$matrix_runner" \
  >/dev/null
if rg -q 'git ls-remote|require_current_reference' "$runner"; then
    fail 'matched physical runner still races a moving remote branch'
fi
rg -F 'pin_policy:"observed-current-then-frozen"' "$runner" >/dev/null
rg -F 'qwen38_validate_physical_matrix_seal "$PHYSICAL_MATRIX_RECEIPT"' \
  "$runner" >/dev/null
rg -F 'qwen38_copy_physical_matrix_seal' \
  "$matrix_runner" >/dev/null
rg -F 'matched_physical_record_semantic_repeat_tokens' "$runner" >/dev/null
rg -F 'HF2Q_DEBUG_TOKENIZE_NO_SPECIAL_TOKENS' \
  "$ROOT_DIR/src/serve/mod.rs" >/dev/null
if rg -n 'event_count' "$runner" "$ROOT_DIR/scripts/qwen38_matched_physical_contract.sh" \
  | rg -q 'total_semantic_completion_tokens|comparison_units_per_second'; then
    fail 'SSE event count leaked into semantic-token measurement'
fi
cohort_line=$(rg -n '^matched_physical_validate_matrix_reference_cohort ' \
  "$matrix_runner" | tail -1 | cut -d: -f1)
publish_line=$(rg -n '^matched_publish_result ' "$matrix_runner" \
  | tail -1 | cut -d: -f1)
((cohort_line < publish_line))
policy_line=$(rg -n 'matched_validate_qwen_frozen_routing_policy_log "\$log"' \
  "$runner" | cut -d: -f1)
((stop_line < policy_line && policy_line < quality_line))
[[ "$(rg -c 'matched_physical_validate_reopened_(child|matrix) "\$OUT_DIR"' \
  "$ROOT_DIR/scripts/qwen38_matched_physical"*.sh \
  | awk -F: '{sum += $2} END {print sum}')" == 2 ]]

echo 'matched physical source contract: pass'
