#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

validate_qwen35_rectangular_power_log() {
    local mode=$1 label=$2
    awk -F '\t' -v mode="$mode" -v label="$label" '
      BEGIN {
        expected[1]=label "-before-launch";
        expected[2]=label "-loaded-warm";
        expected[3]=label "-measurement-start";
        expected[4]=label "-measurement-end";
        expected[5]=label "-after-shutdown";
      }
      NF != 5 || $1 !~ /^[0-9]+$/ || $2 != "ac" || $3 != mode ||
        $4 !~ /^[0-9]+$/ || $5 != expected[NR] || (NR > 1 && $1 < prior) {bad++}
      {count++; codes[$4]=1; prior=$1}
      END {exit !(count == 5 && bad == 0 && length(codes) == 1)}
    '
}

validate_qwen35_rectangular_publication_shape() {
    local publication=$1
    awk -v line="$publication" 'BEGIN {
      if (line !~ /Qwen rectangular prefill published/ || line !~ /lanes=4/ || line !~ /checkpoint_at_end=true/) exit 1
      rows=line; sub(/^.*rows_per_lane=/,"",rows); sub(/ .*/,"",rows); rows += 0
      aggregate=line; sub(/^.*aggregate_rows=/,"",aggregate); sub(/ .*/,"",aggregate); aggregate += 0
      exit !(rows >= 16 && rows <= 128 && aggregate == 4 * rows)
    }'
}

if [[ ${1:-} == --self-test ]]; then
    valid_log=$(printf '1\tac\thigh\t2\toff-a-before-launch\n2\tac\thigh\t2\toff-a-loaded-warm\n3\tac\thigh\t2\toff-a-measurement-start\n4\tac\thigh\t2\toff-a-measurement-end\n5\tac\thigh\t2\toff-a-after-shutdown\n')
    printf '%s\n' "$valid_log" | validate_qwen35_rectangular_power_log high off-a || {
        echo "rectangular policy verifier self-test rejected portable power evidence" >&2
        exit 1
    }
    printf '%s\n' "${valid_log/off-a-measurement-end/off-a-wrong-phase}" \
        | validate_qwen35_rectangular_power_log high off-a && {
        echo "rectangular policy verifier self-test accepted corrupt power evidence" >&2
        exit 1
    }
    valid_publication='Qwen rectangular prefill published lanes=4 rows_per_lane=84 aggregate_rows=336 mtp_prefill=true checkpoint_at_end=true mtp_outcome=Succeeded'
    validate_qwen35_rectangular_publication_shape "$valid_publication" || {
        echo "rectangular policy verifier self-test rejected valid row shape" >&2
        exit 1
    }
    validate_qwen35_rectangular_publication_shape \
        "${valid_publication/aggregate_rows=336/aggregate_rows=335}" && {
        echo "rectangular policy verifier self-test accepted corrupt row shape" >&2
        exit 1
    }
    validate_qwen35_rectangular_publication_shape \
        "${valid_publication/checkpoint_at_end=true/checkpoint_at_end=false}" && {
        echo "rectangular policy verifier self-test accepted an unpublished checkpoint" >&2
        exit 1
    }
    echo "Qwen rectangular policy verifier self-test: PASS"
    exit 0
fi

receipt=${1:?usage: verify_qwen35_rectangular_policy_receipt.sh RECEIPT [SOURCE_ROOT]}
SOURCE_ROOT=${2:-}
for command in awk cmp find git jq perl rg shasum sort stat tail tr wc; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$receipt" == /* && -f "$receipt" && -r "$receipt" && ! -L "$receipt" ]] || {
    echo "receipt must be an absolute readable regular non-symlink" >&2
    exit 2
}
evidence_root=$(cd "$(dirname "$receipt")" && pwd -P)
[[ "$(basename "$receipt")" == receipt.json ]] || {
    echo "rectangular policy receipt must be named receipt.json" >&2
    exit 2
}
tmp_dir=$(mktemp -d "${TMPDIR:-/var/tmp}/qwen-rectangular-verify.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

fail() {
    echo "invalid Qwen rectangular policy receipt: $*" >&2
    exit 1
}

metric_value() {
    local file=$1 name=$2
    awk -v name="$name" '$1 == name {value=$2; found++}
      END {
        if (found != 1 || value !~ /^[0-9]+([.][0-9]+)?$/) exit 1;
        print value
      }' "$file"
}

jq -e '
  .schema == 1 and .verdict == "pass"
  and .gate == "qwen35-rectangular-policy-abba"
  and (.source.commit | test("^[0-9a-f]{40}$"))
  and (.source.sha256 | test("^[0-9a-f]{64}$"))
  and (.model.sha256 | test("^[0-9a-f]{64}$"))
  and (.model.shape == "qwen38-dense" or .model.shape == "qwen36-moe")
  and .workload.process_order == ["off-a","on-a","on-b","off-b"]
  and .workload.same_binary == true
  and .workload.trials_per_process == 5 and .workload.lanes == 4
  and .workload.stable_boundary_rows == {minimum:16,maximum:128}
  and .workload.max_tokens == 2 and .workload.temperature == 0
  and .workload.seed == 42 and .workload.speculation == "auto"
  and .workload.coalesce_us == 25000
  and .workload.kv_cache_budget_bytes == 51539607552
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
  and .thresholds == {min_wave_speedup:1.01,max_single_overhead_ms:50}
  and (.equality.semantic_and_token_sha256 | test("^[0-9a-f]{64}$"))
' "$receipt" >/dev/null || fail "top-level contract"

source_commit=$(jq -er '.source.commit' "$receipt")
binary=$(jq -er '.source.binary' "$receipt")
binary_sha=$(jq -er '.source.sha256' "$receipt")
model_path=$(jq -er '.model.path' "$receipt")
model_sha=$(jq -er '.model.sha256' "$receipt")
model_bytes=$(jq -er '.model.bytes' "$receipt")
model_shape=$(jq -er '.model.shape' "$receipt")
power_mode=$(jq -er '.environment.power_mode' "$receipt")
host_contention_owner_pgid=$(jq -er \
    '.environment.host_contention.owner_pgid' "$receipt")
[[ -x "$binary" && ! -L "$binary" \
    && "$(shasum -a 256 "$binary" | awk '{print $1}')" == "$binary_sha" \
    && -f "$model_path" && ! -L "$model_path" \
    && "$(shasum -a 256 "$model_path" | awk '{print $1}')" == "$model_sha" \
    && "$(stat -f '%z' "$model_path" 2>/dev/null || stat -c '%s' "$model_path")" == "$model_bytes" ]] \
    || fail "binary/model identity"
grep -aFq "$source_commit" "$binary" || fail "binary/source commit binding"
if [[ -z "$SOURCE_ROOT" ]]; then
    SOURCE_ROOT=${binary%/target/release/hf2q}
fi
[[ "$SOURCE_ROOT" == /* && "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$source_commit" \
    && -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] \
    || fail "clean source identity"

case "$model_shape" in
    qwen38-dense) expected_arch=qwen35; expected_mtp=succeeded; expected_mtp_bool=true ;;
    qwen36-moe) expected_arch=qwen35moe; expected_mtp=not-requested; expected_mtp_bool=false ;;
    *) fail "model shape" ;;
esac

for label in off-a on-a on-b off-b; do
    process_dir="$evidence_root/$label"
    summary="$process_dir/summary.json"
    manifest="$process_dir/evidence.sha256"
    [[ -d "$process_dir" && -f "$summary" && -f "$manifest" ]] \
        || fail "$label evidence files"
    (cd "$process_dir" && shasum -a 256 -c evidence.sha256 >/dev/null) \
        || fail "$label raw evidence manifest"
    expected_summary_sha=$(jq -er --arg label "$label" \
        '.evidence.processes[$label].summary_sha256' "$receipt")
    expected_manifest_sha=$(jq -er --arg label "$label" \
        '.evidence.processes[$label].manifest_sha256' "$receipt")
    [[ "$(shasum -a 256 "$summary" | awk '{print $1}')" == "$expected_summary_sha" \
        && "$(shasum -a 256 "$manifest" | awk '{print $1}')" == "$expected_manifest_sha" \
        && "$(jq -er '.evidence_manifest_sha256' "$summary")" == "$expected_manifest_sha" ]] \
        || fail "$label summary/manifest binding"
    arm=${label%%-*}
    jq -e --arg label "$label" --arg arm "$arm" '
      .label == $label and .arm == $arm
      and (.single_engine_ttft_samples_ms | length) == 5
      and (.single_wall_samples_ms | length) == 5
      and (.wave_samples_seconds | length) == 5
      and (.sampled_peak_rss_kib | numbers) > 0
      and .runtime == {clean_environment:true,home:"/var/empty",
        path:"/usr/bin:/bin:/usr/sbin:/sbin",tmpdir:"/var/tmp",
        locale:{LANG:"C",LC_ALL:"C"},rust_backtrace:"1",
        hf2q:{tq_kv:"1",encoder_session:"1",ffn_terminal_k_batch:"8",
          speculation:"auto"},serve:{kv_persist:false,
            kv_cache_budget_bytes:51539607552,cache_dir:"evidence-local"}}
    ' "$summary" >/dev/null || fail "$label summary shape"
    single_ttft_samples=$(jq -Rsc 'split("\n")
      | map(select(length > 0) | tonumber)' "$process_dir/single-ttft-ms") \
        || fail "$label raw engine TTFT samples"
    single_wall_samples=$(jq -Rsc 'split("\n")
      | map(select(length > 0) | tonumber)' "$process_dir/single-wall-ms") \
        || fail "$label raw single wall samples"
    wave_samples=$(jq -Rsc 'split("\n")
      | map(select(length > 0) | tonumber)' "$process_dir/wave-wall-seconds") \
        || fail "$label raw wave samples"
    jq -en --argjson ttft "$single_ttft_samples" \
        --argjson single "$single_wall_samples" --argjson wave "$wave_samples" '
      ($ttft | length) == 5 and ($single | length) == 5 and ($wave | length) == 5
      and all($ttft[]; (. | numbers) > 0)
      and all($single[]; (. | numbers) > 0)
      and all($wave[]; (. | numbers) > 0)
    ' >/dev/null || fail "$label raw timing sample shape"
    single_ttft_median=$(jq -nr --argjson values "$single_ttft_samples" '$values|sort|.[2]')
    single_wall_median=$(jq -nr --argjson values "$single_wall_samples" '$values|sort|.[2]')
    wave_median=$(jq -nr --argjson values "$wave_samples" '$values|sort|.[2]')
    awk 'NF != 1 || $1 !~ /^[1-9][0-9]*$/ {bad++}
      END {exit !(NR > 0 && bad == 0)}' \
        "$process_dir/rss-kib" || fail "$label raw RSS samples"
    sampled_peak_rss=$(sort -n "$process_dir/rss-kib" | tail -1)
    jq -e --argjson ttft "$single_ttft_samples" \
        --argjson single "$single_wall_samples" --argjson wave "$wave_samples" \
        --argjson ttft_median "$single_ttft_median" \
        --argjson single_median "$single_wall_median" \
        --argjson wave_median "$wave_median" --argjson rss "$sampled_peak_rss" '
      .single_engine_ttft_samples_ms == $ttft
      and .single_wall_samples_ms == $single
      and .wave_samples_seconds == $wave
      and .single_median_ttft_ms == $ttft_median
      and .single_median_wall_ms == $single_median
      and .wave_median_seconds == $wave_median
      and .sampled_peak_rss_kib == $rss
    ' "$summary" >/dev/null || fail "$label raw/summary derivation"
    jq -n --arg arm "$arm" --argjson single "$single_wall_samples" \
        --argjson wave "$wave_samples" --argjson single_median "$single_wall_median" \
        --argjson wave_median "$wave_median" '{arm:$arm,
          single_wall_samples_ms:$single,wave_samples_seconds:$wave,
          single_median_wall_ms:$single_median,wave_median_seconds:$wave_median}' \
        >"$tmp_dir/$label-derived.json"
    for item in \
        thermal_settle_sha256:thermal-settle.tsv \
        thermal_measurement_sha256:thermal-measurement.tsv \
        contention_settle_sha256:contention-settle.tsv \
        contention_measurement_sha256:contention-measurement.tsv \
        power_sha256:power.tsv; do
        key=${item%%:*}
        file=${item#*:}
        [[ "$(jq -er --arg key "$key" '.environment[$key]' "$summary")" == \
            "$(shasum -a 256 "$process_dir/$file" | awk '{print $1}')" ]] \
            || fail "$label $file hash"
    done
    thermal_validate_settle_log "$process_dir/thermal-settle.tsv" 60 5 \
        || fail "$label thermal settle"
    host_contention_validate_settle_log "$process_dir/contention-settle.tsv" 60 5 \
        || fail "$label contention settle"
    thermal_validate_fair_or_better_measurement_log \
        "$process_dir/thermal-measurement.tsv" 5 \
        || fail "$label thermal measurement"
    host_contention_validate_measurement_log \
        "$process_dir/contention-measurement.tsv" 5 \
        || fail "$label contention measurement"
    host_contention_validate_thermal_alignment \
        "$process_dir/thermal-measurement.tsv" \
        "$process_dir/contention-measurement.tsv" \
        || fail "$label thermal/contention alignment"
    for contention_log in contention-settle.tsv contention-measurement.tsv; do
        awk -F '\t' -v owner="$host_contention_owner_pgid" '
          NF != 6 || $4 != owner { bad++ }
          END { exit !(NR > 0 && bad == 0) }
        ' "$process_dir/$contention_log" \
            || fail "$label $contention_log owner binding"
    done
    validate_qwen35_rectangular_power_log "$power_mode" "$label" \
        <"$process_dir/power.tsv" || fail "$label AC/power continuity"
    actual_arch=$(jq -er '
      [.data[] | select(.loaded == true)]
      | if length == 1 then .[0].arch else error("one model required") end
    ' "$process_dir/models.json")
    [[ "$actual_arch" == "$expected_arch" ]] || fail "$label architecture"
    expected_admit=false
    expected_coalesce=0
    if [[ "$arm" == on ]]; then
        expected_admit=true
        expected_coalesce=25000
    fi
    EXPECTED_ADMIT="$expected_admit" EXPECTED_COALESCE="$expected_coalesce" \
    EXPECTED_MTP="$expected_mtp_bool" perl -ne '
      if (/Qwen35 SlotAware prefill transaction ceiling selected/) {
        $seen++;
        $admit=$1 if /cross_slot_admit=(true|false)/;
        $coalesce=$1 if /cross_slot_coalesce_us=([0-9]+)/;
        $policy=$1 if /speculation_policy=(Auto|Off)/;
        $mtp=$1 if /mtp_capable=(true|false)/;
      }
      END {exit 1 unless $seen == 1 && $admit eq $ENV{EXPECTED_ADMIT}
        && $coalesce == $ENV{EXPECTED_COALESCE} && $policy eq "Auto"
        && $mtp eq $ENV{EXPECTED_MTP}}
    ' "$process_dir/server.stderr" || fail "$label immutable runtime policy"
    perl -ne '
      if (/resolved serving plan/) {
        $seen++;
        $persist=$1 if /kv_persist_enabled=(true|false)/;
        $budget=$1 if /kv_persist_budget_bytes=([0-9]+)/;
        $cache=$1 if /kv_cache_budget_bytes=([0-9]+)/;
      }
      END {exit 1 unless $seen == 1 && $persist eq "false" && $budget == 0
        && $cache == 51539607552}
    ' "$process_dir/server.stderr" || fail "$label persistence-free serve plan"
    qwen36_reject_fatal_log "$process_dir/server.stderr" \
        || fail "$label fatal log"
    server_command=$(<"$process_dir/server-command.txt")
    [[ " $server_command " == *" --cache-dir $process_dir/runtime-cache "* \
        && " $server_command " != *" --kv-persist "* ]] \
        || fail "$label cache/persistence argv"

    for ((trial = 1; trial <= 5; trial++)); do
        wave_json="$process_dir/waves/$trial.json"
        wave_log="$process_dir/waves/$trial.log"
        single_response="$process_dir/responses/single-$trial.json"
        [[ -f "$wave_json" && -f "$wave_log" && -f "$single_response" ]] \
            || fail "$label trial $trial files"
        timing_files=("$process_dir"/responses/wave-"$trial"-*.timing)
        [[ "${#timing_files[@]}" == 4 ]] \
            || fail "$label trial $trial timing file cardinality"
        awk -F '\t' '
          NF != 2 || $1 !~ /^[0-9]+([.][0-9]+)?$/ ||
            $2 !~ /^[0-9]+([.][0-9]+)?$/ || $2 < $1 {bad++}
          END {exit !(NR == 4 && bad == 0)}
        ' "${timing_files[@]}" || fail "$label trial $trial timing shape"
        launch_skew=$(awk -F '\t' '
          NR == 1 {minimum=$1; maximum=$1}
          $1 < minimum {minimum=$1} $1 > maximum {maximum=$1}
          END {printf "%.9f", maximum-minimum}
        ' "${timing_files[@]}")
        latest_start=$(awk -F '\t' \
          'NR == 1 || $1 > value {value=$1} END {print value}' \
          "${timing_files[@]}")
        earliest_finish=$(awk -F '\t' \
          'NR == 1 || $2 < value {value=$2} END {print value}' \
          "${timing_files[@]}")
        before_metric=$(metric_value "$process_dir/waves/$trial.metrics-before" \
            hf2q_qwen_rectangular_prefill_cohorts_total) \
            || fail "$label trial $trial metric before"
        after_metric=$(metric_value "$process_dir/waves/$trial.metrics-after" \
            hf2q_qwen_rectangular_prefill_cohorts_total) \
            || fail "$label trial $trial metric after"
        metric_delta=$(awk -v before="$before_metric" -v after="$after_metric" \
            'BEGIN {printf "%.0f", after-before}')
        jq -e --argjson skew "$launch_skew" --argjson latest "$latest_start" \
            --argjson earliest "$earliest_finish" --argjson delta "$metric_delta" '
          .launch_skew_seconds == $skew
          and .latest_start == $latest
          and .earliest_finish == $earliest
          and .actual_overlap == ($latest < $earliest)
          and .cohort_metric_delta == $delta
          and $skew <= 0.100 and $latest < $earliest
        ' "$wave_json" >/dev/null || fail "$label trial $trial raw wave derivation"
        single_prompt=$(jq -er '.usage.prompt_tokens' "$single_response")
        [[ "$(jq -er '.usage.prompt_tokens_details.cached_tokens' "$single_response")" == 0 \
            && "$(jq -er '.prompt_tokens' "$wave_json")" == "$single_prompt" ]] \
            || fail "$label trial $trial single eligibility"
        for ((lane = 1; lane <= 4; lane++)); do
            lane_response="$process_dir/responses/wave-$trial-$lane.json"
            [[ "$(jq -er '.usage.prompt_tokens' "$lane_response")" == "$single_prompt" \
                && "$(jq -er '.usage.prompt_tokens_details.cached_tokens' "$lane_response")" == 0 ]] \
                || fail "$label trial $trial lane $lane eligibility"
        done
        publication=$(rg 'Qwen rectangular prefill published' \
            "$wave_log" || true)
        if [[ "$arm" == on ]]; then
            [[ "$(printf '%s\n' "$publication" | awk 'NF {n++} END {print n+0}')" == 1 \
                && "$(jq -er '.cohort_metric_delta' "$wave_json")" == 1 \
                && "$(jq -er '.actual_overlap' "$wave_json")" == true ]] \
                || fail "$label trial $trial rectangular proof"
            validate_qwen35_rectangular_publication_shape "$publication" \
                || fail "$label trial $trial row shape"
            if [[ "$expected_mtp" == succeeded ]]; then
                [[ "$publication" == *"mtp_prefill=true checkpoint_at_end=true mtp_outcome=Succeeded"* ]] \
                    || fail "$label trial $trial MTP outcome"
            else
                [[ "$publication" == *"mtp_prefill=false checkpoint_at_end=true mtp_outcome=NotRequested"* ]] \
                    || fail "$label trial $trial no-MTP outcome"
            fi
        else
            [[ -z "$publication" \
                && "$(jq -er '.cohort_metric_delta' "$wave_json")" == 0 ]] \
                || fail "$label trial $trial OFF isolation"
        fi
    done
done

for item in \
    caffeinate_log_sha256:caffeinate.log \
    assertions_sha256:caffeinate.log.assertions \
    events_baseline_sha256:caffeinate.log.power-events.baseline \
    events_final_sha256:caffeinate.log.power-events.final \
    events_new_sha256:caffeinate.log.power-events.new; do
    key=${item%%:*}
    file=${item#*:}
    [[ -f "$evidence_root/$file" \
        && "$(jq -er --arg key "$key" '.evidence.power_guard[$key]' "$receipt")" == \
            "$(shasum -a 256 "$evidence_root/$file" | awk '{print $1}')" ]] \
        || fail "power guard $file binding"
done
rg -q 'caffeinate' "$evidence_root/caffeinate.log.assertions" \
    || fail "caffeinate assertion evidence"
qwen36_extract_new_power_events \
    "$evidence_root/caffeinate.log.power-events.baseline" \
    "$evidence_root/caffeinate.log.power-events.final" \
    "$tmp_dir/power-events.new" \
    || fail "power guard event derivation"
cmp -s "$tmp_dir/power-events.new" \
    "$evidence_root/caffeinate.log.power-events.new" \
    || fail "power guard event receipt derivation"
[[ ! -s "$evidence_root/caffeinate.log.power-events.new" ]] \
    || fail "power guard observed a disallowed sleep/wake event"

for replica in a b; do
    while IFS= read -r relative; do
        cmp -s "$evidence_root/off-$replica/$relative" \
            "$evidence_root/on-$replica/$relative" \
            || fail "request equality $replica/$relative"
    done < <(cd "$evidence_root/off-$replica" && find requests -type f -print | sort)
done

off_semantic_sha=$(jq -Ssc 'map({
  message:(.choices[0].message | {role,content,reasoning_content,tool_calls,refusal}),
  finish_reason:.choices[0].finish_reason,
  usage:(.usage | {prompt_tokens,completion_tokens,total_tokens,prompt_tokens_details})})' \
  "$evidence_root"/off-*/responses/{single,wave}-*.json \
  | shasum -a 256 | awk '{print $1}')
on_semantic_sha=$(jq -Ssc 'map({
  message:(.choices[0].message | {role,content,reasoning_content,tool_calls,refusal}),
  finish_reason:.choices[0].finish_reason,
  usage:(.usage | {prompt_tokens,completion_tokens,total_tokens,prompt_tokens_details})})' \
  "$evidence_root"/on-*/responses/{single,wave}-*.json \
  | shasum -a 256 | awk '{print $1}')
receipt_semantic_sha=$(jq -er '.equality.semantic_and_token_sha256' "$receipt")
[[ "$off_semantic_sha" == "$on_semantic_sha" \
    && "$on_semantic_sha" == "$receipt_semantic_sha" ]] \
    || fail "canonical semantic equality"

for arm in off on; do
    jq -s --arg arm "$arm" '{arm:$arm,
      single_wall_samples_ms:(map(.single_wall_samples_ms)|add),
      wave_samples_seconds:(map(.wave_samples_seconds)|add)}
      | def median: sort as $s | ($s|length) as $n
          | if ($n % 2) == 1 then $s[($n/2)|floor]
            else (($s[$n/2-1]+$s[$n/2])/2) end;
        .single_median_wall_ms=(.single_wall_samples_ms|median)
      | .wave_median_seconds=(.wave_samples_seconds|median)' \
        "$tmp_dir/$arm-a-derived.json" \
        "$tmp_dir/$arm-b-derived.json" >"$tmp_dir/$arm-summary.json"
    cmp -s "$tmp_dir/$arm-summary.json" "$evidence_root/$arm-summary.json" \
        || fail "$arm aggregate derivation"
    expected_sha=$(jq -er --arg arm "$arm" '.evidence.aggregates[$arm + "_sha256"]' \
        "$receipt")
    [[ "$(shasum -a 256 "$evidence_root/$arm-summary.json" | awk '{print $1}')" == "$expected_sha" ]] \
        || fail "$arm aggregate hash"
done

: >"$tmp_dir/single-overhead-ms"
for replica in a b; do
    for ((trial = 1; trial <= 5; trial++)); do
        off_wall=$(tr -d '[:space:]' \
            <"$evidence_root/off-$replica/responses/single-$trial.wall")
        on_wall=$(tr -d '[:space:]' \
            <"$evidence_root/on-$replica/responses/single-$trial.wall")
        awk -v off="$off_wall" -v on="$on_wall" \
            'BEGIN {printf "%.6f\n", (on-off)*1000}' >>"$tmp_dir/single-overhead-ms"
    done
done
cmp -s "$tmp_dir/single-overhead-ms" "$evidence_root/single-overhead-ms" \
    || fail "matched single overhead derivation"
wave_speedup=$(jq -nr --slurpfile off "$tmp_dir/off-summary.json" \
    --slurpfile on "$tmp_dir/on-summary.json" \
    '$off[0].wave_median_seconds/$on[0].wave_median_seconds')
neighbor_a=$(jq -nr --slurpfile off "$tmp_dir/off-a-derived.json" \
    --slurpfile on "$tmp_dir/on-a-derived.json" \
    '$off[0].wave_median_seconds/$on[0].wave_median_seconds')
neighbor_b=$(jq -nr --slurpfile off "$tmp_dir/off-b-derived.json" \
    --slurpfile on "$tmp_dir/on-b-derived.json" \
    '$off[0].wave_median_seconds/$on[0].wave_median_seconds')
median_overhead=$(jq -nr --slurpfile off "$tmp_dir/off-summary.json" \
    --slurpfile on "$tmp_dir/on-summary.json" \
    '$on[0].single_median_wall_ms-$off[0].single_median_wall_ms')
max_overhead=$(sort -n "$tmp_dir/single-overhead-ms" | tail -1)
jq -e --argjson wave "$wave_speedup" --argjson a "$neighbor_a" \
    --argjson b "$neighbor_b" --argjson median "$median_overhead" \
    --argjson maximum "$max_overhead" \
    --argjson samples "$(jq -Rsc 'split("\n")|map(select(length>0)|tonumber)' "$tmp_dir/single-overhead-ms")" '
      .result.wave_speedup == $wave
      and .result.neighboring_process_speedups == [$a,$b]
      and .result.single_median_overhead_ms == $median
      and .result.single_max_matched_overhead_ms == $maximum
      and .result.single_matched_overhead_samples_ms == $samples
      and $wave >= 1.01 and $a > 1 and $b > 1 and $maximum <= 50
    ' "$receipt" >/dev/null || fail "derived verdict"

echo "Qwen rectangular policy receipt: VERIFIED"
