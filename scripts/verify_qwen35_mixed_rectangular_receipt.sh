#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/agentic_cache_lifecycle_contract.sh
source "$script_dir/agentic_cache_lifecycle_contract.sh"
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"
# shellcheck source=scripts/qwen35_mixed_rectangular_contract.sh
source "$script_dir/qwen35_mixed_rectangular_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

qwen35_mixed_checked_decoder_canonical_json() {
  jq -e '
    select(
      (.message.content | type) == "string" and (.message.content | length) > 0
      and .message.tool_calls == []
      and (.finish_reason == "stop" or .finish_reason == "length")
      and (.usage.prompt_tokens | numbers) > 0
      and ((.usage.prompt_tokens_details.cached_tokens // 0) | numbers) >= 0
    )
  '
}

if [[ ${1:-} == --self-test ]]; then
  tmp=$(mktemp -d "${TMPDIR:-/var/tmp}/qwen-mixed-self-test.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT
  cat >"$tmp/frames.tsv" <<'EOF'
1.000000000	{"choices":[{"delta":{"content":"a"},"finish_reason":null}]}
1.100000000	{"choices":[{"delta":{"content":"b"},"finish_reason":null}]}
1.200000000	{"choices":[{"delta":{"content":"c"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":0}}}
1.300000000	[DONE]
EOF
  qwen35_mixed_semantic_trace_json "$tmp/frames.tsv" 0.990 1.050 1.150 \
    >"$tmp/trace.json"
  qwen35_mixed_validate_semantic_trace "$tmp/trace.json" 3 100 101
  if qwen35_mixed_validate_semantic_trace "$tmp/trace.json" 3 100 99; then
    echo "mixed verifier self-test accepted an over-bound semantic gap" >&2
    exit 1
  fi
  jq '.semantic_after_wave = 0' "$tmp/trace.json" >"$tmp/no-after.json"
  if qwen35_mixed_validate_semantic_trace "$tmp/no-after.json" 3 100 101; then
    echo "mixed verifier self-test accepted a decoder with no post-wave event" >&2
    exit 1
  fi
  valid_publication='Qwen rectangular prefill published lanes=4 rows_per_lane=84 aggregate_rows=336 mtp_prefill=true checkpoint_at_end=true mtp_outcome=Succeeded'
  qwen35_mixed_validate_publication "$valid_publication" succeeded
  if qwen35_mixed_validate_publication \
    "${valid_publication/aggregate_rows=336/aggregate_rows=335}" succeeded; then
    echo "mixed verifier self-test accepted a corrupt rectangular shape" >&2
    exit 1
  fi
  if qwen35_mixed_validate_publication \
    "${valid_publication/checkpoint_at_end=true/checkpoint_at_end=false}" succeeded; then
    echo "mixed verifier self-test accepted an unpublished checkpoint" >&2
    exit 1
  fi
  valid_power=$(printf '1\tac\thigh\t2\ton-a-before-launch\n2\tac\thigh\t2\ton-a-loaded-warm\n3\tac\thigh\t2\ton-a-measurement-start\n4\tac\thigh\t2\ton-a-measurement-end\n5\tac\thigh\t2\ton-a-after-shutdown\n')
  printf '%s\n' "$valid_power" | qwen35_mixed_validate_power_log high on-a
  if printf '%s\n' "${valid_power/on-a-measurement-end/on-a-wrong}" \
    | qwen35_mixed_validate_power_log high on-a; then
    echo "mixed verifier self-test accepted a corrupt power phase" >&2
    exit 1
  fi
  jq -n '{message:{content:"decoder",tool_calls:[]},finish_reason:"stop",
    usage:{prompt_tokens:10,prompt_tokens_details:{cached_tokens:0}}}' \
    >"$tmp/decoder-canonical.json"
  qwen35_mixed_checked_decoder_canonical_json \
    <"$tmp/decoder-canonical.json" >"$tmp/decoder-checked.json"
  jq -e 'type == "object" and .message.content == "decoder"' \
    "$tmp/decoder-checked.json" >/dev/null
  jq '.usage.prompt_tokens = 0' "$tmp/decoder-canonical.json" \
    >"$tmp/decoder-invalid.json"
  if qwen35_mixed_checked_decoder_canonical_json \
    <"$tmp/decoder-invalid.json" >"$tmp/decoder-invalid-checked.json"; then
    echo "mixed verifier self-test accepted a zero-token decoder" >&2
    exit 1
  fi
  echo "Qwen mixed rectangular verifier self-test: PASS"
  exit 0
fi

receipt=${1:?usage: verify_qwen35_mixed_rectangular_receipt.sh RECEIPT [SOURCE_ROOT]}
SOURCE_ROOT=${2:-}
for command in awk basename cat cmp find git grep jq mktemp perl rg sed shasum \
  sort stat tail tr wc; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done
[[ "$receipt" == /* && -f "$receipt" && -r "$receipt" && ! -L "$receipt" \
  && "$(basename "$receipt")" == receipt.json ]] || exit 2
root=$(cd "$(dirname "$receipt")" && pwd -P)
tmp=$(mktemp -d "${TMPDIR:-/var/tmp}/qwen-mixed-verify.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
fail() { echo "invalid Qwen mixed rectangular receipt: $*" >&2; exit 1; }

median_file() {
  local file=$1 count
  count=$(wc -l <"$file" | tr -d ' ')
  sort -n "$file" | awk -v count="$count" '
    NR == int((count + 1) / 2) {left=$1}
    NR == int((count + 2) / 2) {right=$1}
    END {if (count % 2) print left; else print (left + right) / 2}
  '
}

jq -e '
  .schema == 1 and .verdict == "pass"
  and .gate == "qwen35-mixed-rectangular-cell"
  and (.source.commit | test("^[0-9a-f]{40}$"))
  and (.source.sha256 | test("^[0-9a-f]{64}$"))
  and (.model.sha256 | test("^[0-9a-f]{64}$"))
  and (.model.shape == "qwen38-dense" or .model.shape == "qwen36-moe")
  and .workload == {process_order:["off-a","on-a","on-b","off-b"],
    same_binary:true,trials_per_process:5,max_slots:8,live_decoders:1,
    cold_prefills:4,prefill_max_tokens:2,decoder_max_tokens:512,
    temperature:0,seed:42,speculation:"auto",coalesce_us:25000,
    kv_cache_budget_bytes:51539607552}
  and .environment.power == "ac"
  and .environment.thermal == "nominal-settle-and-fair-or-better-measurement"
  and .environment.host_contention.policy == "process-group-cpu-v2"
  and .environment.host_contention.maximum_foreign_cpu_percent == 100
  and .environment.host_contention.owner_scope == "release-gate-process-group"
  and (.environment.host_contention.owner_pgid | numbers) > 0
  and .environment.host_contention.owner_pgid
    == (.environment.host_contention.owner_pgid | floor)
  and .environment.host_contention.continuous == true
  and .environment.clean_process_environment == true
  and .environment.serve_kv_persist == false
  and .thresholds == {min_mixed_speedup:1.01,min_semantic_events:3,
    max_decoder_ttft_ms:15000,max_semantic_gap_ms:15000,
    max_prefill_tail_ms:60000,max_launch_skew_ms:100}
  and (.equality.canonical_sha256 | test("^[0-9a-f]{64}$"))
' "$receipt" >/dev/null || fail top-level-contract

source_commit=$(jq -er .source.commit "$receipt")
binary=$(jq -er .source.binary "$receipt")
binary_sha=$(jq -er .source.sha256 "$receipt")
model=$(jq -er .model.path "$receipt")
model_sha=$(jq -er .model.sha256 "$receipt")
model_bytes=$(jq -er .model.bytes "$receipt")
model_snapshot=$(jq -er .model.snapshot "$receipt")
model_shape=$(jq -er .model.shape "$receipt")
power_mode=$(jq -er .environment.power_mode "$receipt")
host_contention_owner_pgid=$(jq -er \
  '.environment.host_contention.owner_pgid' "$receipt")
[[ -x "$binary" && ! -L "$binary" \
  && "$(shasum -a 256 "$binary" | awk '{print $1}')" == "$binary_sha" \
  && -f "$model" && ! -L "$model" \
  && "$(shasum -a 256 "$model" | awk '{print $1}')" == "$model_sha" \
  && "$(stat -f '%z' "$model" 2>/dev/null || stat -c '%s' "$model")" == "$model_bytes" \
  && "$(stat -f '%d:%i:%z:%m:%c' "$model" 2>/dev/null \
    || stat -c '%d:%i:%s:%Y:%Z' "$model")" == "$model_snapshot" ]] \
  || fail binary-model-identity
grep -aFq "$source_commit" "$binary" || fail binary-source-binding
if [[ -z "$SOURCE_ROOT" ]]; then SOURCE_ROOT=${binary%/target/release/hf2q}; fi
[[ "$SOURCE_ROOT" == /* && "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$source_commit" \
  && -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] \
  || fail clean-source-identity
case "$model_shape" in
  qwen38-dense) expected_arch=qwen35; expected_mtp=succeeded; expected_mtp_bool=true ;;
  qwen36-moe) expected_arch=qwen35moe; expected_mtp=not-requested; expected_mtp_bool=false ;;
  *) fail model-shape ;;
esac

for label in off-a on-a on-b off-b; do
  arm=${label%%-*}
  process="$root/$label"
  summary="$process/summary.json"
  manifest="$process/evidence.sha256"
  [[ -d "$process" && -f "$summary" && -f "$manifest" ]] || fail "$label files"
  (cd "$process" && shasum -a 256 -c evidence.sha256 >/dev/null) \
    || fail "$label raw-manifest"
  [[ "$(shasum -a 256 "$summary" | awk '{print $1}')" == \
      "$(jq -er --arg label "$label" '.evidence.processes[$label].summary_sha256' "$receipt")" \
    && "$(shasum -a 256 "$manifest" | awk '{print $1}')" == \
      "$(jq -er --arg label "$label" '.evidence.processes[$label].manifest_sha256' "$receipt")" \
    && "$(jq -er .evidence.manifest_sha256 "$summary")" == \
      "$(shasum -a 256 "$manifest" | awk '{print $1}')" ]] || fail "$label bindings"
  jq -e --arg label "$label" --arg arm "$arm" '
    .label == $label and .arm == $arm
    and .runtime == {clean_environment:true,max_slots:8,
      scheduler:"inflight-batched",speculation:"auto",
      kv_cache_budget_bytes:51539607552,kv_persist:false,
      cache_dir:"evidence-local"}
    and (.sampled_peak_rss_kib | numbers) > 0
    and (.server_pid | numbers) > 0 and .server_pid == (.server_pid | floor)
    and (.evidence.execution_sha256 | test("^[0-9a-f]{64}$"))
    and (.evidence.canonical_sha256 | test("^[0-9a-f]{64}$"))
  ' "$summary" >/dev/null || fail "$label summary-contract"

  for item in thermal_settle_sha256:thermal-settle.tsv \
    thermal_measurement_sha256:thermal-measurement.tsv \
    contention_settle_sha256:contention-settle.tsv \
    contention_measurement_sha256:contention-measurement.tsv \
    power_sha256:power.tsv execution_sha256:execution.json \
    canonical_sha256:canonical.jsonl; do
    key=${item%%:*}; file=${item#*:}
    [[ "$(jq -er --arg key "$key" '.evidence[$key]' "$summary")" == \
      "$(shasum -a 256 "$process/$file" | awk '{print $1}')" ]] \
      || fail "$label $file hash"
  done
  thermal_validate_settle_log "$process/thermal-settle.tsv" 60 5 \
    || fail "$label thermal-settle"
  host_contention_validate_settle_log "$process/contention-settle.tsv" 60 5 \
    || fail "$label contention-settle"
  thermal_validate_fair_or_better_measurement_log "$process/thermal-measurement.tsv" 5 \
    || fail "$label thermal-measurement"
  host_contention_validate_measurement_log "$process/contention-measurement.tsv" 5 \
    || fail "$label contention-measurement"
  host_contention_validate_thermal_alignment "$process/thermal-measurement.tsv" \
    "$process/contention-measurement.tsv" || fail "$label host-alignment"
  for contention_log in contention-settle.tsv contention-measurement.tsv; do
    awk -F '\t' -v owner="$host_contention_owner_pgid" '
      NF != 6 || $4 != owner { bad++ }
      END { exit !(NR > 0 && bad == 0) }
    ' "$process/$contention_log" || fail "$label $contention_log owner binding"
  done
  qwen35_mixed_validate_power_log "$power_mode" "$label" <"$process/power.tsv" \
    || fail "$label power"
  actual_model=$(jq -er '[.data[] | select(.loaded == true)]
    | if length == 1 then .[0] else error("one model") end' "$process/models.json")
  [[ "$(jq -r .arch <<<"$actual_model")" == "$expected_arch" \
    && "$(jq -r .id <<<"$actual_model")" == "$(jq -er .model_id "$summary")" ]] \
    || fail "$label model-identity"
  [[ "$(tr -d '[:space:]' <"$process/server-pid.txt")" == \
    "$(jq -er .server_pid "$summary")" ]] || fail "$label PID-binding"
  server_command=$(<"$process/server-command.txt")
  [[ " $server_command " == *" --model $model "* \
    && " $server_command " == *" --cache-dir $process/runtime-cache "* \
    && " $server_command " == *" --scheduler inflight-batched "* \
    && " $server_command " == *" --max-slots 8 "* \
    && " $server_command " != *" --kv-persist "* ]] || fail "$label server-command"
  expected_admit=false; expected_coalesce=0
  [[ "$arm" == on ]] && { expected_admit=true; expected_coalesce=25000; }
  EXPECTED_ADMIT="$expected_admit" EXPECTED_COALESCE="$expected_coalesce" \
    EXPECTED_MTP="$expected_mtp_bool" perl -ne '
      if (/Qwen35 SlotAware prefill transaction ceiling selected/) {
        $seen++; $admit=$1 if /cross_slot_admit=(true|false)/;
        $coalesce=$1 if /cross_slot_coalesce_us=([0-9]+)/;
        $policy=$1 if /speculation_policy=(Auto|Off)/;
        $mtp=$1 if /mtp_capable=(true|false)/;
      }
      END {exit 1 unless $seen == 1 && $admit eq $ENV{EXPECTED_ADMIT}
        && $coalesce == $ENV{EXPECTED_COALESCE} && $policy eq "Auto"
        && $mtp eq $ENV{EXPECTED_MTP}}
    ' "$process/server.stderr" || fail "$label worker-policy"
  perl -ne '
    if (/resolved serving plan/) {
      $seen++; $persist=$1 if /kv_persist_enabled=(true|false)/;
      $budget=$1 if /kv_persist_budget_bytes=([0-9]+)/;
      $cache=$1 if /kv_cache_budget_bytes=([0-9]+)/;
    }
    END {exit 1 unless $seen == 1 && $persist eq "false" && $budget == 0
      && $cache == 51539607552}
  ' "$process/server.stderr" || fail "$label serve-plan"
  qwen36_reject_fatal_log "$process/server.stderr" || fail "$label fatal-log"

  : >"$tmp/$label-canonical.jsonl"
  : >"$tmp/$label-prefill-tail-ms"
  : >"$tmp/$label-semantic-ttft-ms"
  : >"$tmp/$label-semantic-gap-ms"
  for ((trial = 1; trial <= 5; trial++)); do
    frames="$process/responses/decoder-$trial.frames.tsv"
    sse="$process/responses/decoder-$trial.sse"
    events="$tmp/$label-decoder-$trial.events.jsonl"
    qwen36_extract_and_validate_sse "$label decoder $trial" "$sse" "$events" \
      || fail "$label trial $trial SSE"
    jq -se '
      ([.[] | .choices[0].delta.role // empty] == ["assistant"])
      and ([.[] | .choices[0].delta.tool_calls[]?] | length) == 0
      and ([.[] | .choices[0].finish_reason // empty] | length) == 1
      and ([ .[] | select(has("usage")) ] | length) == 1
    ' "$events" >/dev/null || fail "$label trial $trial SSE-contract"
    sed -n 's/^data: //p' "$sse" >"$tmp/$label-$trial-sse-payloads"
    awk -F '\t' '{print substr($0, index($0, $2))}' "$frames" \
      >"$tmp/$label-$trial-frame-payloads"
    cmp -s "$tmp/$label-$trial-sse-payloads" "$tmp/$label-$trial-frame-payloads" \
      || fail "$label trial $trial timestamped-SSE-binding"
    jq -e --argjson trial "$trial" '
      .stream == true and .stream_options == {include_usage:true}
      and .temperature == 0 and .seed == 42 and .max_tokens == 512
      and .repetition_penalty == 1 and .hf2q_enable_thinking == false
      and .chat_template_kwargs == {enable_thinking:false}
      and ((.tools // []) | length) == 0
      and (.messages | length) == 2
      and (.messages[1].content
        | contains("Trial " + ($trial | tostring) + ". Begin with STREAM_BEGIN."))
    ' "$process/requests/decoder-$trial.json" >/dev/null \
      || fail "$label trial $trial decoder-request"
    request_started=$(tr -d '[:space:]' <"$process/responses/decoder-$trial.started")
    decoder_timing="$process/responses/decoder-$trial.timing"
    awk -F '\t' -v started="$request_started" '
      NF != 2 || $1 != started || $1 !~ /^[0-9]+([.][0-9]+)?$/ ||
        $2 !~ /^[0-9]+([.][0-9]+)?$/ || $2 < $1 {bad++}
      END {exit !(NR == 1 && bad == 0)}
    ' "$decoder_timing" || fail "$label trial $trial decoder-timing"
    timing_files=("$process"/responses/prefill-"$trial"-*.timing)
    [[ "${#timing_files[@]}" == 4 ]] || fail "$label trial $trial timing-cardinality"
    awk -F '\t' 'NF != 2 || $1 !~ /^[0-9]+([.][0-9]+)?$/ ||
      $2 !~ /^[0-9]+([.][0-9]+)?$/ || $2 < $1 {bad++}
      END {exit !(NR == 4 && bad == 0)}' "${timing_files[@]}" \
      || fail "$label trial $trial timing-shape"
    earliest_start=$(awk -F '\t' 'NR == 1 || $1 < value {value=$1} END {print value}' \
      "${timing_files[@]}")
    latest_start=$(awk -F '\t' 'NR == 1 || $1 > value {value=$1} END {print value}' \
      "${timing_files[@]}")
    earliest_finish=$(awk -F '\t' 'NR == 1 || $2 < value {value=$2} END {print value}' \
      "${timing_files[@]}")
    latest_finish=$(awk -F '\t' 'NR == 1 || $2 > value {value=$2} END {print value}' \
      "${timing_files[@]}")
    launch_skew=$(awk -v first="$earliest_start" -v last="$latest_start" \
      'BEGIN {printf "%.9f", last-first}')
    prefill_tail=$(awk -v first="$earliest_start" -v last="$latest_finish" \
      'BEGIN {printf "%.6f", (last-first)*1000}')
    qwen35_mixed_semantic_trace_json "$frames" "$request_started" \
      "$earliest_start" "$latest_finish" >"$tmp/$label-$trial-trace.json" \
      || fail "$label trial $trial trace-derivation"
    qwen35_mixed_validate_semantic_trace "$tmp/$label-$trial-trace.json" 3 15000 15000 \
      || fail "$label trial $trial starvation"
    decoder_finished=$(awk -F '\t' '{print $2}' "$decoder_timing")
    jq -e --argjson decoder_finished "$decoder_finished" \
      '.terminal_at <= $decoder_finished' "$tmp/$label-$trial-trace.json" >/dev/null \
      || fail "$label trial $trial decoder-terminal-clock"
    cmp -s "$tmp/$label-$trial-trace.json" "$process/waves/$trial.semantic.json" \
      || fail "$label trial $trial raw-semantic-binding"
    jq -er '.first_semantic_ms' "$tmp/$label-$trial-trace.json" \
      >>"$tmp/$label-semantic-ttft-ms"
    jq -er '.max_semantic_gap_ms' "$tmp/$label-$trial-trace.json" \
      >>"$tmp/$label-semantic-gap-ms"
    printf '%s\n' "$prefill_tail" >>"$tmp/$label-prefill-tail-ms"
    before=$(qwen35_mixed_metric_value "$process/waves/$trial.metrics-before" \
      hf2q_qwen_rectangular_prefill_cohorts_total) \
      || fail "$label trial $trial metric-before"
    after=$(qwen35_mixed_metric_value "$process/waves/$trial.metrics-after" \
      hf2q_qwen_rectangular_prefill_cohorts_total) \
      || fail "$label trial $trial metric-after"
    delta=$(awk -v before="$before" -v after="$after" 'BEGIN {printf "%.0f", after-before}')
    jq -e --argjson launch_skew "$launch_skew" \
      --argjson earliest_start "$earliest_start" --argjson latest_start "$latest_start" \
      --argjson earliest_finish "$earliest_finish" --argjson latest_finish "$latest_finish" \
      --argjson tail "$prefill_tail" --argjson delta "$delta" \
      --slurpfile semantic "$tmp/$label-$trial-trace.json" '
      .launch_skew_seconds == $launch_skew
      and .earliest_start == $earliest_start and .latest_start == $latest_start
      and .earliest_finish == $earliest_finish and .latest_finish == $latest_finish
      and .actual_prefill_overlap == ($latest_start < $earliest_finish)
      and .prefill_tail_ms == $tail and .cohort_metric_delta == $delta
      and .semantic == $semantic[0]
      and ($launch_skew * 1000) <= 100 and $latest_start < $earliest_finish
      and $tail <= 60000
    ' "$process/waves/$trial.json" >/dev/null || fail "$label trial $trial wave-derivation"
    publication=$(rg 'Qwen rectangular prefill published' \
      "$process/waves/$trial.log" || true)
    if [[ "$arm" == on ]]; then
      [[ "$delta" == 1 \
        && "$(printf '%s\n' "$publication" | awk 'NF {n++} END {print n+0}')" == 1 ]] \
        || fail "$label trial $trial rectangular-cardinality"
      qwen35_mixed_validate_publication "$publication" "$expected_mtp" \
        || fail "$label trial $trial publication"
    else
      [[ "$delta" == 0 && -z "$publication" ]] || fail "$label trial $trial OFF-isolation"
    fi
    jq -e '
      (.choices | length) == 1
      and (.usage.prompt_tokens | numbers) > 0
      and (.usage.prompt_tokens_details.cached_tokens // 0) == 0
      and (.usage.completion_tokens | numbers) > 0
    ' "$process/responses/prefill-$trial-1.json" >/dev/null \
      || fail "$label trial $trial cold-prefill"
    prompt_tokens=$(jq -er .usage.prompt_tokens \
      "$process/responses/prefill-$trial-1.json")
    for ((lane = 1; lane <= 4; lane++)); do
      jq -e --argjson trial "$trial" --argjson lane "$lane" '
        .stream == false and .temperature == 0 and .seed == 42
        and .max_tokens == 2 and .repetition_penalty == 1
        and .hf2q_enable_thinking == false
        and .chat_template_kwargs == {enable_thinking:false}
        and (.messages | length) == 1 and .messages[0].role == "user"
        and (.messages[0].content
          | startswith("mixed trial-" + ($trial|tostring)
            + " lane-" + ($lane|tostring) + ". "))
        and ([.messages[0].content | scan("cache ")] | length) == 64
        and (.messages[0].content | endswith("Return exactly OK."))
      ' "$process/requests/prefill-$trial-$lane.json" >/dev/null \
        || fail "$label trial $trial lane $lane prefill-request"
      response="$process/responses/prefill-$trial-$lane.json"
      [[ "$(jq -er .usage.prompt_tokens "$response")" == "$prompt_tokens" \
        && "$(jq -er '.usage.prompt_tokens_details.cached_tokens // 0' "$response")" == 0 ]] \
        || fail "$label trial $trial lane $lane eligibility"
      qwen35_mixed_canonical_unary_json "$response" \
        | jq -c --argjson trial "$trial" --argjson lane "$lane" \
          '. + {kind:"prefill",trial:$trial,lane:$lane}' \
        >>"$tmp/$label-prefill-$trial.rows"
    done
    qwen35_mixed_canonical_sse_json "$events" \
      | qwen35_mixed_checked_decoder_canonical_json \
        >"$tmp/$label-decoder-$trial-canonical.json" \
      || fail "$label trial $trial decoder-canonical"
    jq -c --argjson trial "$trial" '. + {kind:"decoder",trial:$trial}' \
      "$tmp/$label-decoder-$trial-canonical.json" >>"$tmp/$label-canonical.jsonl"
    cat "$tmp/$label-prefill-$trial.rows" >>"$tmp/$label-canonical.jsonl"
    rm -f "$tmp/$label-prefill-$trial.rows"
  done
  cmp -s "$tmp/$label-prefill-tail-ms" "$process/prefill-tail-ms" \
    || fail "$label raw-prefill-tail-derivation"
  cmp -s "$tmp/$label-semantic-ttft-ms" "$process/semantic-ttft-ms" \
    || fail "$label raw-semantic-ttft-derivation"
  cmp -s "$tmp/$label-semantic-gap-ms" "$process/semantic-gap-ms" \
    || fail "$label raw-semantic-gap-derivation"
  cmp -s "$tmp/$label-canonical.jsonl" "$process/canonical.jsonl" \
    || fail "$label canonical-derivation"

  : >"$tmp/$label-execution.rows"
  while IFS= read -r relative; do
    phase=${relative#responses/}; phase=${phase%.headers}; stream=false
    [[ "$phase" == decoder-* ]] && stream=true
    agentic_lifecycle_execution_receipt_json "$process/$relative" \
      "$model_sha" qwen35 "$expected_arch" \
      | jq -c --arg phase "$phase" --argjson stream "$stream" \
        '. + {phase:$phase,stream:$stream}' >>"$tmp/$label-execution.rows" \
      || fail "$label $phase execution-header"
  done < <(cd "$process" && find responses -name '*.headers' -type f -print | sort)
  jq -s . "$tmp/$label-execution.rows" >"$tmp/$label-execution.json"
  cmp -s "$tmp/$label-execution.json" "$process/execution.json" \
    || fail "$label execution-derivation"
  jq -e '
    length == 33
    and ([.[].pool_key_b64] | unique | length) == 1
    and ([.[].generation] | unique | length) == 1
    and ([.[] | select(.stream == true)] | length) == 5
  ' "$tmp/$label-execution.json" >/dev/null || fail "$label execution-cardinality"

  for raw in prefill-tail-ms semantic-ttft-ms semantic-gap-ms; do
    awk 'NF != 1 || $1 !~ /^[0-9]+([.][0-9]+)?$/ {bad++}
      END {exit !(NR == 5 && bad == 0)}' "$process/$raw" || fail "$label $raw"
  done
  awk 'NF != 1 || $1 !~ /^[1-9][0-9]*$/ {bad++}
    END {exit !(NR > 0 && bad == 0)}' "$process/rss-kib" || fail "$label RSS"
  prefill_median=$(median_file "$process/prefill-tail-ms")
  max_prefill=$(sort -n "$process/prefill-tail-ms" | tail -1)
  max_ttft=$(sort -n "$process/semantic-ttft-ms" | tail -1)
  max_gap=$(sort -n "$process/semantic-gap-ms" | tail -1)
  peak_rss=$(sort -n "$process/rss-kib" | tail -1)
  jq -e --argjson prefill_median "$prefill_median" \
    --argjson max_prefill "$max_prefill" --argjson max_ttft "$max_ttft" \
    --argjson max_gap "$max_gap" --argjson peak_rss "$peak_rss" '
    .prefill_median_ms == $prefill_median
    and .max_prefill_tail_ms == $max_prefill
    and .max_semantic_ttft_ms == $max_ttft
    and .max_semantic_gap_ms == $max_gap
    and .sampled_peak_rss_kib == $peak_rss
    and $max_prefill <= 60000 and $max_ttft <= 15000 and $max_gap <= 15000
  ' "$summary" >/dev/null || fail "$label summary-derivation"
done

for item in caffeinate_log_sha256:caffeinate.log \
  assertions_sha256:caffeinate.log.assertions \
  events_baseline_sha256:caffeinate.log.power-events.baseline \
  events_final_sha256:caffeinate.log.power-events.final \
  events_new_sha256:caffeinate.log.power-events.new; do
  key=${item%%:*}; file=${item#*:}
  [[ "$(jq -er --arg key "$key" '.evidence.power_guard[$key]' "$receipt")" == \
    "$(shasum -a 256 "$root/$file" | awk '{print $1}')" ]] || fail "power guard $file"
done
rg -q caffeinate "$root/caffeinate.log.assertions" || fail power-assertion
qwen36_extract_new_power_events "$root/caffeinate.log.power-events.baseline" \
  "$root/caffeinate.log.power-events.final" "$tmp/power-events.new" \
  || fail power-event-derivation
cmp -s "$tmp/power-events.new" "$root/caffeinate.log.power-events.new" \
  || fail power-event-binding
[[ ! -s "$tmp/power-events.new" ]] || fail sleep-wake-event

for replica in a b; do
  while IFS= read -r relative; do
    cmp -s "$root/off-$replica/$relative" "$root/on-$replica/$relative" \
      || fail "request equality $replica/$relative"
  done < <(cd "$root/off-$replica" && find requests -type f -print | sort)
  cmp -s "$root/off-$replica/canonical.jsonl" "$root/on-$replica/canonical.jsonl" \
    || fail "canonical equality $replica"
done
cmp -s "$root/off-a/canonical.jsonl" "$root/off-b/canonical.jsonl" \
  || fail canonical-replica-equality
[[ "$(shasum -a 256 "$root/on-a/canonical.jsonl" | awk '{print $1}')" == \
  "$(jq -er .equality.canonical_sha256 "$receipt")" ]] || fail canonical-receipt

cat "$root/off-a/prefill-tail-ms" "$root/off-b/prefill-tail-ms" >"$tmp/off"
cat "$root/on-a/prefill-tail-ms" "$root/on-b/prefill-tail-ms" >"$tmp/on"
off_median=$(median_file "$tmp/off")
on_median=$(median_file "$tmp/on")
speedup=$(awk -v off="$off_median" -v on="$on_median" 'BEGIN {print off/on}')
neighbor_a=$(awk -v off="$(median_file "$root/off-a/prefill-tail-ms")" \
  -v on="$(median_file "$root/on-a/prefill-tail-ms")" 'BEGIN {print off/on}')
neighbor_b=$(awk -v off="$(median_file "$root/off-b/prefill-tail-ms")" \
  -v on="$(median_file "$root/on-b/prefill-tail-ms")" 'BEGIN {print off/on}')
jq -e --argjson speedup "$speedup" --argjson a "$neighbor_a" \
  --argjson b "$neighbor_b" --argjson off "$off_median" --argjson on "$on_median" '
  .result.mixed_prefill_speedup == $speedup
  and .result.neighboring_process_speedups == [$a,$b]
  and .result.off_median_prefill_tail_ms == $off
  and .result.on_median_prefill_tail_ms == $on
  and $speedup >= 1.01 and $a > 1 and $b > 1
' "$receipt" >/dev/null || fail derived-verdict

echo "Qwen mixed rectangular receipt: VERIFIED"
