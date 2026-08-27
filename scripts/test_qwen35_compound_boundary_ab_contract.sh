#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner="$script_dir/bench_qwen35_compound_boundary_ab.sh"
wave_contract="$script_dir/qwen35_compound_wave_contract.sh"

fail() {
    echo "$*" >&2
    exit 1
}

bash -n "$runner"
[[ -f "$wave_contract" ]] || fail 'matched compound A/B wave contract is missing'
bash -n "$wave_contract"
# shellcheck source=scripts/qwen35_compound_wave_contract.sh
source "$wave_contract"
grep -Fq 'source "$script_dir/qwen36_watchdog_validate.sh"' "$runner" \
    || fail 'matched compound A/B does not use the shared PID watchdog'
[[ "$(grep -Fc 'qwen36_bind_server_process "http://$HOST:$PORT"' "$runner")" -ge 5 ]] \
    || fail 'matched compound A/B does not bind readiness and measured waves to the spawned PID'
grep -Fq 'qwen36_reject_fatal_log "$engine_dir/server.stderr"' "$runner" \
    || fail 'matched compound A/B does not reject fatal server logs before receipt creation'
grep -Fq '[[ "$MODEL" == /* && -f "$MODEL" && -r "$MODEL" && ! -L "$MODEL" ]]' "$runner" \
    || fail 'matched compound A/B accepts mutable or ambiguous model paths'
grep -Fq 'model_snapshot=$(hf2q_release_model_snapshot "$MODEL")' "$runner" \
    || fail 'matched compound A/B does not snapshot exact model file identity'
[[ "$(grep -Fc 'assert_model_unchanged' "$runner")" -ge 8 ]] \
    || fail 'matched compound A/B does not guard the model across both engine runs and waves'
grep -Fq -- '--arg model_snapshot "$model_snapshot"' "$runner" \
    || fail 'matched compound A/B receipt omits its model snapshot'
grep -Fq 'prove_compound_wave "$label" "$trial"' "$runner" \
    || fail 'matched compound A/B does not apply the arm-aware wave contract'
grep -Fq 'run_engine baseline-a "$BASELINE_BIN" "$baseline_commit" "$baseline_dependency" 0' "$runner" \
    || fail 'baseline must not require candidate-era physical telemetry'
grep -Fq 'run_engine candidate-a "$CANDIDATE_BIN" "$candidate_commit" "$candidate_dependency" 1' "$runner" \
    || fail 'candidate must require physical instrumentation'
grep -Fq 'readonly MIN_SINGLE_TTFT_RATIO=1.01' "$runner" \
    || fail 'single TTFT acceptance policy is externally weakenable'
grep -Fq 'readonly MIN_SINGLE_WALL_RATIO=1.0' "$runner" \
    || fail 'single wall acceptance policy is externally weakenable'
grep -Fq 'readonly MIN_FOUR_SLOT_WAVE_RATIO=1.0' "$runner" \
    || fail 'four-slot wave acceptance policy is externally weakenable'
for fixed in \
    'readonly TRIALS=5' \
    'readonly PROMPT_LINES=80' \
    'readonly MAX_TOKENS=8' \
    'readonly MAX_SLOTS=4' \
    'readonly KV_CACHE_BUDGET_BYTES=51539607552'; do
    grep -Fq "$fixed" "$runner" \
        || fail "matched compound A/B policy is externally weakenable: $fixed"
done
grep -Fq 'qwen35_compound_require_fresh_out_dir "$OUT_DIR"' "$runner" \
    || fail 'matched compound A/B does not reject stale evidence directories'
grep -Fq 'qwen35_compound_publish_receipt "$receipt_tmp" "$OUT_DIR/receipt.json"' "$runner" \
    || fail 'matched compound A/B does not publish its receipt atomically'
grep -Fq 'run_engine baseline-a ' "$runner" \
    || fail 'ABBA baseline-a process is missing'
grep -Fq 'run_engine candidate-a ' "$runner" \
    || fail 'ABBA candidate-a process is missing'
grep -Fq 'run_engine candidate-b ' "$runner" \
    || fail 'ABBA candidate-b process is missing'
grep -Fq 'run_engine baseline-b ' "$runner" \
    || fail 'ABBA baseline-b process is missing'
grep -Fq 'single_ttft_samples_ms:$single_ttft_samples_ms' "$runner" \
    || fail 'receipt omits raw single TTFT samples'
grep -Fq 'four_slot_wave_samples_seconds:$four_slot_wave_samples_seconds' "$runner" \
    || fail 'receipt omits raw four-slot wave samples'
grep -Fq 'qwen35_compound_aggregate_arm baseline ' "$runner" \
    || fail 'ABBA process summaries are not aggregated by the tested contract'

tmp_dir=$(mktemp -d "${TMPDIR:-/var/tmp}/hf2q-compound-wave-contract.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

write_metrics() {
    local output=$1 steps=$2 handles=$3 forwards=$4 body=$5 head=$6 width=$7
    printf '%s\n' \
        "hf2q_qwen_decode_scheduler_steps_total $steps" \
        "hf2q_qwen_decode_scheduler_handles_total $handles" \
        "hf2q_qwen_decode_ordinary_target_forwards_total $forwards" \
        "hf2q_qwen_decode_ordinary_target_body_rows_total $body" \
        "hf2q_qwen_decode_ordinary_target_head_rows_total $head" \
        "hf2q_qwen_decode_scheduler_max_width $width" \
        "hf2q_qwen_decode_ordinary_target_body_max_width $width" \
        "hf2q_qwen_decode_ordinary_target_head_max_width $width" >"$output"
}

printf '%s\n' 'hf2q_uptime_seconds 1' >"$tmp_dir/absent.before"
printf '%s\n' 'hf2q_uptime_seconds 2' >"$tmp_dir/absent.after"
prove_compound_wave baseline 1 "$tmp_dir/absent.before" "$tmp_dir/absent.after" \
    "$tmp_dir/absent.json" 0 4
jq -e '
  .client_wave_complete == true and .client_count == 4
  and .physical_instrumentation == "unavailable"
  and .physical_width_four_observed == null
' "$tmp_dir/absent.json" >/dev/null \
    || fail 'uninstrumented baseline receipt is not explicit'
if prove_compound_wave candidate 1 "$tmp_dir/absent.before" "$tmp_dir/absent.after" \
    "$tmp_dir/should-not-exist.json" 1 4 2>/dev/null; then
    fail 'candidate accepted missing physical telemetry'
fi

write_metrics "$tmp_dir/width4.before" 10 10 10 10 10 1
write_metrics "$tmp_dir/width4.after" 12 18 12 18 18 4
prove_compound_wave candidate 2 "$tmp_dir/width4.before" "$tmp_dir/width4.after" \
    "$tmp_dir/width4.json" 1 4
jq -e '
  .physical_instrumentation == "available"
  and .physical_width_four_observed == true
  and .scheduler == {steps_delta:2,handles_delta:8,max_width:4}
  and .target == {forwards_delta:2,body_rows_delta:8,head_rows_delta:8,
                  body_max_width:4,head_max_width:4}
' "$tmp_dir/width4.json" >/dev/null \
    || fail 'candidate width-four receipt lost physical evidence'

write_metrics "$tmp_dir/width1.before" 10 10 10 10 10 1
write_metrics "$tmp_dir/width1.after" 18 18 18 18 18 1
prove_compound_wave baseline 3 "$tmp_dir/width1.before" "$tmp_dir/width1.after" \
    "$tmp_dir/width1.json" 0 4
jq -e '.physical_width_four_observed == false' "$tmp_dir/width1.json" >/dev/null \
    || fail 'instrumented scalar baseline was mislabeled'
prove_compound_wave candidate 3 "$tmp_dir/width1.before" "$tmp_dir/width1.after" \
    "$tmp_dir/candidate-width1.json" 1 4
jq -e '
  .physical_instrumentation == "available"
  and .physical_width_four_observed == false
' "$tmp_dir/candidate-width1.json" >/dev/null \
    || fail 'one-token candidate wave did not honestly record scalar execution'

sed '/ordinary_target_head_rows_total/d' "$tmp_dir/width4.after" \
    >"$tmp_dir/partial.after"
if prove_compound_wave baseline 4 "$tmp_dir/width4.before" "$tmp_dir/partial.after" \
    "$tmp_dir/should-not-exist.json" 0 4 2>/dev/null; then
    fail 'partial physical telemetry was treated as unavailable'
fi

printf '%s\n' 'hf2q_qwen_decode_scheduler_steps_total 99' \
    >>"$tmp_dir/width4.after"
if prove_compound_wave candidate 5 "$tmp_dir/width4.before" "$tmp_dir/width4.after" \
    "$tmp_dir/should-not-exist.json" 1 4 2>/dev/null; then
    fail 'duplicate physical telemetry was accepted'
fi
if prove_compound_wave baseline 6 "$tmp_dir/absent.before" "$tmp_dir/absent.after" \
    "$tmp_dir/should-not-exist.json" 0 3 2>/dev/null; then
    fail 'incomplete client wave was accepted'
fi

qwen35_compound_validate_policy 1.01 1.0 1.0
if qwen35_compound_validate_policy 0.1 0.1 0.1 2>/dev/null; then
    fail 'weakened acceptance policy was accepted'
fi

fresh_dir="$tmp_dir/fresh-evidence"
qwen35_compound_require_fresh_out_dir "$fresh_dir"
[[ -d "$fresh_dir" ]] || fail 'fresh evidence directory was not created'
printf '%s\n' '{"verdict":"pass"}' >"$fresh_dir/stale-receipt.json"
if qwen35_compound_require_fresh_out_dir "$fresh_dir" 2>/dev/null; then
    fail 'stale evidence directory was accepted'
fi

publish_dir="$tmp_dir/publish"
mkdir -p "$publish_dir"
printf '%s\n' '{"schema":3,"verdict":"pass"}' >"$publish_dir/receipt.tmp"
qwen35_compound_publish_receipt \
    "$publish_dir/receipt.tmp" "$publish_dir/receipt.json"
[[ -f "$publish_dir/receipt.json" && ! -e "$publish_dir/receipt.tmp" ]] \
    || fail 'receipt publication was not atomic'
printf '%s\n' '{"schema":3,"verdict":"pass"}' >"$publish_dir/second.tmp"
if qwen35_compound_publish_receipt \
    "$publish_dir/second.tmp" "$publish_dir/receipt.json" 2>/dev/null; then
    fail 'receipt publication overwrote existing evidence'
fi

for run in a b; do
    jq -n --arg label "baseline-$run" --argjson offset "$([[ "$run" == a ]] && printf 0 || printf 2)" '{
      label:$label,binary:"/exact/hf2q",binary_sha256:"abc",
      source_commit:"0123456789012345678901234567890123456789",
      dependency_identity:"mlx@exact",model_id:"model",
      compound_receipts:0,
      single_ttft_samples_ms:[1 + $offset, 5 + $offset],
      single_wall_samples_seconds:[2 + $offset, 6 + $offset],
      four_slot_wave_samples_seconds:[3 + $offset, 7 + $offset],
      wave_execution_receipts:[{trial:1,client_wave_complete:true,client_count:4}]
    }' >"$tmp_dir/aggregate-$run.json"
done
qwen35_compound_aggregate_arm baseline "$tmp_dir/aggregate-a.json" \
    "$tmp_dir/aggregate-b.json" "$tmp_dir/aggregate.json"
jq -e '
  .process_order == ["baseline-a","baseline-b"]
  and .single_ttft_samples_ms == [1,5,3,7]
  and .single_median_ttft_ms == 4
  and .single_median_wall_seconds == 5
  and .four_slot_median_wave_seconds == 6
  and (.wave_execution_receipts | length) == 2
' "$tmp_dir/aggregate.json" >/dev/null \
    || fail 'ABBA aggregation lost raw samples or computed the wrong median'
jq '.binary_sha256 = "drift"' "$tmp_dir/aggregate-b.json" \
    >"$tmp_dir/aggregate-drift.json"
if qwen35_compound_aggregate_arm baseline "$tmp_dir/aggregate-a.json" \
    "$tmp_dir/aggregate-drift.json" "$tmp_dir/should-not-exist.json" 2>/dev/null; then
    fail 'ABBA aggregation accepted process identity drift'
fi

grep -Fq 'scripts/qwen35_compound_wave_contract.sh' "$script_dir/../.github/workflows/ci.yml" \
    || fail 'compound wave contract lacks CI syntax coverage'
grep -Fq 'bash scripts/test_qwen35_compound_boundary_ab_contract.sh' \
    "$script_dir/../.github/workflows/ci.yml" \
    || fail 'compound A/B contract fixture is not executed in CI'

echo 'Qwen compound A/B harness contract: PASS'
