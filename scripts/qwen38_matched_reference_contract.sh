#!/usr/bin/env bash

# Pure, model-free predicates shared by the matched Qwen3.8 runner and its
# hosted contract test. The caller owns `set -euo pipefail`.

matched_require_port_available() {
    local port=$1

    [[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || return 2
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null \
      | sed -n '2p' | rg -q .; then
        echo "server listener already occupies port: $port" >&2
        return 1
    fi
}

# Record one thermal, contention, and host-power observation on the same
# timestamp. Every step propagates failure explicitly because callers invoke
# this helper from conditional monitor expressions, where Bash disables the
# usual `errexit` behavior for the complete function body.
matched_record_calibration_observation() {
    local thermal_log=$1 host_log=$2 contention_log=$3 phase=$4
    local owner_pid=$5 owned_server_pid=${6:-} live_power_mode_code

    require_ac_power || return 1
    live_power_mode_code=$(read_live_power_mode_code) || return 1
    # Assigned by each runner's measured power-mode preflight.
    # shellcheck disable=SC2154
    [[ "$live_power_mode_code" == "$power_mode_code" ]] || {
        echo "numeric power-mode canary changed during calibration" >&2
        return 1
    }
    thermal_sample "$thermal_log" "$phase" || return 1
    host_contention_sample "$contention_log" "$phase" "$owner_pid" \
      "$THERMAL_SAMPLED_AT" "$owned_server_pid" || return 1
    # Assigned by the same preflight as power_mode_code.
    # shellcheck disable=SC2154
    printf '%s\tac\t%s\t%s\t%s\t%s\n' "$THERMAL_SAMPLED_AT" \
      "$HOST_CONTENTION_STATE" "$power_mode_name" "$power_mode_code" \
      "$phase" >>"$host_log" || return 1
}

matched_validate_launch_settings() {
    local settings_path=$1 expected_mvn=$2 expected_mv_ext=$3 expected_q5k=$4
    local expected_hf2q_speculation=$5 expected_reference_speculation=$6

    [[ "$expected_mvn" =~ ^[01]$ && "$expected_mv_ext" =~ ^[01]$ \
      && "$expected_q5k" =~ ^[01]$ ]] || return 1
    jq -e --argjson expected_mvn "$expected_mvn" \
      --argjson expected_mv_ext "$expected_mv_ext" \
      --argjson expected_q5k "$expected_q5k" \
      --arg hf2q_speculation "$expected_hf2q_speculation" \
      --arg reference_speculation "$expected_reference_speculation" \
      --arg hf2q_kv "$QWEN38_MATCHED_HF2Q_KV_CACHE" \
      --arg reference_k "$QWEN38_MATCHED_REFERENCE_KV_CACHE_K" \
      --arg reference_v "$QWEN38_MATCHED_REFERENCE_KV_CACHE_V" \
      --argjson hf2q_kv_budget "$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES" \
      --argjson context_tokens "$QWEN38_MATCHED_CONTEXT_TOKENS" '
      .schema == 2
      and .hf2q.dense_decode_mvn == $expected_mvn
      and .hf2q.dense_decode_mv_ext == $expected_mv_ext
      and .hf2q.dense_q5k_canonical_q4x4 == $expected_q5k
      and .hf2q.speculation == $hf2q_speculation
      and .reference.speculation == $reference_speculation
      and .hf2q.kv_cache == $hf2q_kv
      and .reference.kv_cache_k == $reference_k
      and .reference.kv_cache_v == $reference_v
      and .hf2q.kv_cache_budget_bytes == $hf2q_kv_budget
      and .hf2q.context_tokens_per_slot == $context_tokens
      and .reference.context_tokens_total == $context_tokens
    ' "$settings_path" >/dev/null
}

# Re-open a completed hf2q server log and bind the requested dense policy to
# the model's one frozen effective-policy receipt. Requested environment alone
# is not authority: missing, duplicate, malformed, or conflicting load lines
# all fail closed.
matched_validate_qwen_frozen_routing_policy_log() {
    local log_path=$1 expected_mvn=$2 expected_mv_ext=$3 expected_q5k=$4
    local expected_mvn_bool=false expected_mv_ext_bool=false expected_q5k_bool=false

    [[ -f "$log_path" && -r "$log_path" && ! -L "$log_path" \
      && "$expected_mvn" =~ ^[01]$ && "$expected_mv_ext" =~ ^[01]$ \
      && "$expected_q5k" =~ ^[01]$ ]] || return 1
    [[ "$expected_mvn" == 1 ]] && expected_mvn_bool=true
    [[ "$expected_mv_ext" == 1 ]] && expected_mv_ext_bool=true
    [[ "$expected_q5k" == 1 ]] && expected_q5k_bool=true
    EXPECTED_MVN="$expected_mvn_bool" \
    EXPECTED_MV_EXT="$expected_mv_ext_bool" \
    EXPECTED_Q5K="$expected_q5k_bool" perl -ne '
      if (/frozen Qwen GGML routing policy/) {
        $seen++;
        @mvn = /\bdense_decode_mvn=(true|false)\b/g;
        @mv_ext = /\bdense_decode_mv_ext=(true|false)\b/g;
        @q5k = /\bdense_q5k_canonical_q4x4=(true|false)\b/g;
        $valid = 0 unless @mvn == 1 && @mv_ext == 1 && @q5k == 1;
        $actual_mvn = $mvn[0] if @mvn == 1;
        $actual_mv_ext = $mv_ext[0] if @mv_ext == 1;
        $actual_q5k = $q5k[0] if @q5k == 1;
      }
      BEGIN { $valid = 1 }
      END {
        exit 1 unless $valid && $seen == 1
          && $actual_mvn eq $ENV{EXPECTED_MVN}
          && $actual_mv_ext eq $ENV{EXPECTED_MV_EXT}
          && $actual_q5k eq $ENV{EXPECTED_Q5K};
      }
    ' "$log_path"
}

matched_resolve_hf2q_model_id() {
    local models_json=$1
    jq -er '
      [.data[] | select(.loaded == true)]
      | if length == 1 then .[0].id
        else error("expected exactly one loaded hf2q model") end
    ' "$models_json"
}

matched_validate_reference_model_alias() {
    local models_json=$1
    local expected=$2
    jq -e --arg expected "$expected" '
      [.data[]
        | select((.status? == null) or (.status.value == "loaded"))
        | select(.id == $expected
            or any((.aliases // [])[]; . == $expected))]
      | length == 1
    ' "$models_json" >/dev/null
}

matched_parse_sse_stream() {
    local started_at=$1
    local stream_path=$2
    local receipt_path=$3
    local expected=$4
    perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -MJSON::PP -e '
      use strict;
      use warnings;
      my ($started, $stream_path, $receipt_path, $expected) = @ARGV;
      open my $stream, ">", $stream_path or die "open stream: $!";
      my ($first_semantic_ms, $content, $done_count, $event_count) =
        (undef, "", 0, 0);
      while (my $line = <STDIN>) {
        print {$stream} $line or die "write stream: $!";
        next unless $line =~ /^data:\s?(.*?)[\r\n]*$/;
        my $payload = $1;
        if ($payload eq "[DONE]") {
          $done_count++;
          next;
        }
        my $event = eval { decode_json($payload) };
        die "malformed SSE JSON: $@" if $@;
        $event_count++;
        my $choices = $event->{choices};
        next unless ref($choices) eq "ARRAY" && @{$choices} == 1;
        my $delta = $choices->[0]{delta};
        next unless ref($delta) eq "HASH";
        my $semantic = 0;
        for my $field ("reasoning_content", "content") {
          if (defined($delta->{$field}) && !ref($delta->{$field})
              && length($delta->{$field}) > 0) {
            $semantic = 1;
            $content .= $delta->{$field} if $field eq "content";
          }
        }
        if (ref($delta->{tool_calls}) eq "ARRAY" && @{$delta->{tool_calls}} > 0) {
          $semantic = 1;
        }
        if ($semantic && !defined($first_semantic_ms)) {
          $first_semantic_ms =
            (clock_gettime(CLOCK_MONOTONIC) - $started) * 1000.0;
        }
      }
      close $stream or die "close stream: $!";
      die "stream omitted semantic output" unless defined($first_semantic_ms);
      die "stream did not terminate exactly once" unless $done_count == 1;
      die "streamed content mismatch" unless $content eq $expected;
      open my $receipt, ">", $receipt_path or die "open receipt: $!";
      print {$receipt} JSON::PP->new->canonical->encode({
        schema => 1,
        first_semantic_ms => 0.0 + $first_semantic_ms,
        content => $content,
        done_count => $done_count,
        event_count => $event_count,
      }), "\n" or die "write receipt: $!";
      close $receipt or die "close receipt: $!";
    ' "$started_at" "$stream_path" "$receipt_path" "$expected"
}

matched_validate_common_response() {
    local response=$1
    jq -e '
      (.choices | length == 1)
      and .choices[0].message.role == "assistant"
      and (.choices[0].message.content | type == "string" and length > 0)
      and ((.choices[0].message.tool_calls // null) == null)
      and ((.choices[0].message.reasoning_content // null) == null)
      and ((.choices[0].message.refusal // null) == null)
      and (.choices[0].finish_reason | type == "string" and length > 0)
      and (.usage.prompt_tokens | type == "number" and floor == . and . > 0)
      and (.usage.completion_tokens | type == "number" and floor == . and . > 0)
    ' "$response" >/dev/null
}

# Extract one complete Rust source file from a measured response. The coding
# prompts request raw source, but accepting one outer `rust` fence keeps the
# validator focused on program correctness instead of presentation. Prose or
# nested fences remain in the source and fail compilation.
matched_extract_rust_source() {
    local response=$1
    local source_path=$2
    local finish_reason

    finish_reason=$(jq -er '.choices[0].finish_reason' "$response") \
      || return 1
    [[ "$finish_reason" == stop ]] || {
        echo "Rust response did not finish naturally: $finish_reason" >&2
        return 1
    }
    jq -er '.choices[0].message.content' "$response" \
      | perl -0777 -e '
          use strict;
          use warnings;
          my $source = do { local $/; <STDIN> };
          $source =~ s/\r\n/\n/g;
          $source =~ s/^\s*```(?:rust)?[ \t]*\n//;
          $source =~ s/\n```[ \t]*\n?\s*$//;
          die "nested or unmatched Markdown fence\n" if $source =~ /```/;
          die "empty Rust source\n" unless $source =~ /\S/;
          print $source;
        ' >"$source_path"
}

# Compile a model-produced source file with fixed evaluator-owned tests, then
# execute the resulting Rust test binary. This proves complete syntax and the
# requested behavior without requiring two correct engines to choose identical
# low-margin prose or identifier trajectories.
matched_validate_rust_case() {
    local name=$1
    local source_path=$2
    local validation_dir=$3
    local contract_source="$validation_dir/$name-contract.rs"
    local test_binary="$validation_dir/$name-test"
    local compile_log="$validation_dir/$name-rustc.log"
    local test_log="$validation_dir/$name-test.log"
    local authored_test_count authored_assertion_count evaluator_test

    mkdir -p "$validation_dir"
    authored_test_count=$(awk '
      {
        line = $0
        while (match(line, /#[[:space:]]*\[[[:space:]]*test[[:space:]]*\]/)) {
          count++
          line = substr(line, RSTART + RLENGTH)
        }
      }
      END { print count + 0 }
    ' "$source_path")
    [[ "$authored_test_count" == 1 ]] || {
        echo "$name must contain exactly one model-authored #[test]" >&2
        return 1
    }
    authored_assertion_count=$(perl -0777 -ne '
      my $count = () = /\b(?:debug_)?assert(?:_eq|_ne)?\s*!/g;
      print "$count\n";
    ' "$source_path") || return 1
    [[ "$authored_assertion_count" == 1 ]] || {
        echo "$name must contain exactly one model-authored assertion" >&2
        return 1
    }
    cp "$source_path" "$contract_source"
    case "$name" in
        code-a)
            evaluator_test=hf2q_contract_tests::evaluator_fibonacci_contract
            cat >>"$contract_source" <<'RUST'

#[cfg(test)]
mod hf2q_contract_tests {
    use super::*;
    #[test]
    fn evaluator_fibonacci_contract() {
        ::std::assert_eq!(fibonacci(0), 0);
        ::std::assert_eq!(fibonacci(1), 1);
        ::std::assert_eq!(fibonacci(2), 1);
        ::std::assert_eq!(fibonacci(10), 55);
    }
}
RUST
            ;;
        code-b)
            evaluator_test=hf2q_contract_tests::evaluator_binary_search_contract
            cat >>"$contract_source" <<'RUST'

#[cfg(test)]
mod hf2q_contract_tests {
    use super::*;
    #[test]
    fn evaluator_binary_search_contract() {
        let values = [1, 3, 5, 7];
        ::std::assert_eq!(binary_search(&values, 1), Some(0));
        ::std::assert_eq!(binary_search(&values, 5), Some(2));
        ::std::assert_eq!(binary_search(&values, 2), None);
        ::std::assert_eq!(binary_search(&[], 9), None);
    }
}
RUST
            ;;
        code-c)
            evaluator_test=hf2q_contract_tests::evaluator_gcd_contract
            cat >>"$contract_source" <<'RUST'

#[cfg(test)]
mod hf2q_contract_tests {
    use super::*;
    #[test]
    fn evaluator_gcd_contract() {
        ::std::assert_eq!(gcd(48, 18), 6);
        ::std::assert_eq!(gcd(54, 24), 6);
        ::std::assert_eq!(gcd(0, 7), 7);
        ::std::assert_eq!(gcd(13, 13), 13);
    }
}
RUST
            ;;
        *)
            echo "unknown Rust quality case: $name" >&2
            return 1
            ;;
    esac

    if ! rustc --edition 2021 --test --crate-name hf2q_qwen38_eval \
      "$contract_source" -o "$test_binary" >"$compile_log" 2>&1; then
        cat "$compile_log" >&2
        return 1
    fi
    # Execute only the evaluator-owned test. A model-authored `#[test]` may be
    # malformed, hang, or call process::exit(0); none of those may short-circuit
    # the independent behavioral proof. POSIX alarm state survives exec, giving
    # this hosted-safe test a portable bound on macOS without GNU `timeout`.
    if ! perl -e '
      use strict;
      use warnings;
      my $seconds = shift @ARGV;
      alarm $seconds;
      exec @ARGV or die "exec evaluator: $!";
    ' 30 "$test_binary" --quiet --exact "$evaluator_test" \
      >"$test_log" 2>&1; then
        cat "$test_log" >&2
        rm -f -- "$test_binary"
        return 1
    fi
    grep -Eq 'test result: ok\. 1 passed; 0 failed; 0 ignored;' "$test_log" || {
        echo "$name evaluator did not report exactly one passing test" >&2
        cat "$test_log" >&2
        rm -f -- "$test_binary"
        return 1
    }
    rm -f -- "$test_binary"
    jq -n --arg case "$name" --arg evaluator_test "$evaluator_test" \
      --arg source_sha256 "$(shasum -a 256 "$source_path" | awk '{print $1}')" \
      '{schema:1,case:$case,complete_rust:true,compiled:true,
        model_unit_test_present:true,model_assertion_count:1,
        evaluator_test:$evaluator_test,evaluator_tests_passed:true,
        source_sha256:$source_sha256}' \
          >"$validation_dir/$name-quality.json"
}

matched_validate_hf2q_telemetry() {
    local response=$1
    jq -e '
      (.usage.prompt_tokens_details.cached_tokens
        | type == "number" and floor == . and . >= 0)
      and (.x_hf2q_timing.prefill_time_secs | type == "number" and . > 0)
      and (.x_hf2q_timing.decode_time_secs | type == "number" and . > 0)
      and (.x_hf2q_timing.time_to_first_token_ms | type == "number" and . > 0)
      and (.x_hf2q_timing.prefill_tokens_per_sec | type == "number" and . > 0)
      and (.x_hf2q_timing.decode_tokens_per_sec | type == "number" and . > 0)
    ' "$response" >/dev/null
}

matched_validate_reference_telemetry() {
    local response=$1
    jq -e '
      (.timings.cache_n | type == "number" and floor == . and . >= 0)
      and (.timings.prompt_n | type == "number" and floor == . and . > 0)
      and (.timings.predicted_n | type == "number" and floor == . and . > 0)
      and (.timings.prompt_ms | type == "number" and . > 0)
      and (.timings.predicted_ms | type == "number" and . > 0)
      and (.timings.prompt_per_second | type == "number" and . > 0)
      and (.timings.predicted_per_second | type == "number" and . > 0)
    ' "$response" >/dev/null
}

matched_reference_speculation_totals() {
    local rows_file=$1
    local trial=$2
    local totals
    local drafted accepted
    totals=$(jq -sr --argjson trial "$trial" '
      [.[] | select(.engine == "reference" and .trial == $trial)] as $rows
      | [([$rows[] | (.drafted_tokens // 0)] | add // 0),
         ([$rows[] | (.accepted_draft_tokens // 0)] | add // 0)]
      | @tsv
    ' "$rows_file") || return 1
    IFS=$'\t' read -r drafted accepted <<<"$totals"
    [[ "$drafted" =~ ^[0-9]+$ && "$accepted" =~ ^[0-9]+$ ]] || return 1
    ((drafted > 0 && accepted > 0)) || return 1
    printf '%s\t%s\n' "$drafted" "$accepted"
}

matched_validate_host_observation_log() {
    local host_log=$1
    local minimum_samples=$2
    local required_duration=$3
    local maximum_gap=$4
    local stats
    local samples duration gaps invalid starts ends modes

    stats=$(awk -F '\t' -v maximum="$maximum_gap" '
      BEGIN { invalid = 0; gaps = 0 }
      {
        if (NF != 6 || $1 !~ /^[0-9]+$/ || $2 != "ac" || $3 != "quiet" ||
          ($4 != "automatic" && $4 != "high") || $5 !~ /^[0-9]+$/ ||
          length($6) == 0) {
          invalid++
          next
        }
        samples++
        if (samples == 1) first = $1
        if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
        previous = $1
        last = $1
        phases[$6]++
        power_modes[$4 "\t" $5]++
      }
      END {
        duration = samples > 0 ? last - first : -1
        for (mode in power_modes) modes++
        printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\n", samples, duration, gaps,
          invalid, phases["measurement-start"], phases["measurement-end"], modes
      }
    ' "$host_log") || return 1
    IFS=$'\t' read -r samples duration gaps invalid starts ends modes <<<"$stats"
    ((samples >= minimum_samples && duration >= required_duration \
      && gaps == 0 && invalid == 0 && modes == 1)) || return 1
    if ((minimum_samples >= 3)); then
        ((starts == 1 && ends == 1)) || return 1
        awk -F '\t' '$6 == "measurement" { found = 1 }
          END { exit !found }' "$host_log"
    fi
}

# Parse the active AC Energy Mode from `system_profiler SPPowerDataType`.
# The explicit labels are authoritative; the private numeric `pmset` value is
# captured separately only as a cheap continuity canary during measurement.
matched_parse_ac_power_mode() {
    awk '
      /^[[:space:]]*AC Power:[[:space:]]*$/ { in_ac = 1; next }
      in_ac && /^[[:space:]]*Battery Power:[[:space:]]*$/ { in_ac = 0 }
      in_ac && /^[[:space:]]*Current Power Source:[[:space:]]*Yes[[:space:]]*$/ {
        current = 1
      }
      in_ac && /^[[:space:]]*High Power Mode:/ {
        high = $0
        sub(/^.*High Power Mode:[[:space:]]*/, "", high)
      }
      in_ac && /^[[:space:]]*Low Power Mode:/ {
        low = $0
        sub(/^.*Low Power Mode:[[:space:]]*/, "", low)
      }
      END {
        if (!current) exit 1
        if (high == "Yes" && low == "No") print "high"
        else if (high == "No" && low == "No") print "automatic"
        else if (high == "No" && low == "Yes") print "low"
        else exit 1
      }
    '
}

matched_parse_live_power_mode_code() {
    awk '
      $1 == "powermode" && $2 ~ /^[0-9]+$/ && NF == 2 {
        value = $2
        matches++
      }
      END {
        if (matches != 1) exit 1
        print value
      }
    '
}

# Parse the public power-source banner from a complete `pmset -g batt`
# capture. Callers capture the command output before invoking this parser so
# `set -o pipefail` cannot turn an early-match reader into a producer-SIGPIPE
# false negative. Unknown or duplicate banners fail closed.
matched_parse_live_power_source() {
    awk '
      /^Now drawing from '\''AC Power'\''$/ { source = "ac"; matches++ }
      /^Now drawing from '\''Battery Power'\''$/ { source = "battery"; matches++ }
      END {
        if (matches != 1) exit 1
        print source
      }
    '
}

matched_validate_calibration_alignment() {
    local thermal_log=$1
    local host_log=$2
    cmp -s \
      <(awk -F '\t' 'NF == 3 { print $1 "\t" $3 }' "$thermal_log") \
      <(awk -F '\t' 'NF == 6 { print $1 "\t" $6 }' "$host_log")
}

matched_validate_contention_preflight_log() {
    local log_path=$1 expected_rows=$2 expected_owner_pgid=$3
    [[ -f "$log_path" && ! -L "$log_path" \
      && "$expected_rows" =~ ^[1-9][0-9]*$ \
      && "$expected_owner_pgid" =~ ^[1-9][0-9]*$ ]] || return 1
    awk -F '\t' -v rows="$expected_rows" -v owner="$expected_owner_pgid" \
      -v maximum="$HOST_CONTENTION_MAX_FOREIGN_CPU_PERCENT" '
      BEGIN { invalid = 0 }
      {
        if (NF != 6 || $1 !~ /^[0-9]+$/ || $2 != "quiet" \
            || $3 != "preflight" || $4 != owner \
            || $5 !~ /^[0-9]+([.][0-9]+)?$/ || $5 + 0 >= maximum + 0 \
            || $6 != "-") invalid++
        count++
      }
      END { exit !(count == rows && invalid == 0) }
    ' "$log_path"
}

matched_require_evidence_manifest_entry() {
    local child_dir=$1 evidence_path=$2 relative_path
    [[ "$evidence_path" == "$child_dir"/* ]] || return 1
    relative_path=${evidence_path#"$child_dir"/}
    awk -v expected="$relative_path" '
      $2 == expected { matches++ }
      END { exit !(matches == 1) }
    ' "$child_dir/evidence.sha256"
}

matched_validate_reopened_trial_calibration() {
    local trial_dir=$1 expected_owner_pgid=$2 settle_seconds=$3 maximum_gap=$4
    local child_dir=$5
    local thermal_settle="$trial_dir/thermal-settle.tsv"
    local host_settle="$trial_dir/host-settle.tsv"
    local contention_settle="$trial_dir/contention-settle.tsv"
    local thermal_measurement="$trial_dir/thermal-measurement.tsv"
    local host_measurement="$trial_dir/host-measurement.tsv"
    local contention_measurement="$trial_dir/contention-measurement.tsv"
    local path

    [[ -d "$trial_dir" && ! -L "$trial_dir" \
      && ! -e "$trial_dir/calibration-failure.txt" \
      && "$expected_owner_pgid" =~ ^[1-9][0-9]*$ ]] || return 1
    for path in "$thermal_settle" "$host_settle" "$contention_settle" \
      "$thermal_measurement" "$host_measurement" "$contention_measurement"; do
        [[ -f "$path" && -r "$path" && ! -L "$path" ]] || return 1
        matched_require_evidence_manifest_entry "$child_dir" "$path" \
          || return 1
    done

    thermal_validate_settle_log "$thermal_settle" "$settle_seconds" \
      "$maximum_gap" || return 1
    ((THERMAL_LOG_NON_NOMINAL_SAMPLES == 0)) || return 1
    matched_validate_host_observation_log "$host_settle" 2 \
      "$settle_seconds" "$maximum_gap" || return 1
    matched_validate_calibration_alignment "$thermal_settle" "$host_settle" \
      || return 1
    host_contention_validate_settle_log "$contention_settle" \
      "$settle_seconds" "$maximum_gap" || return 1
    ((HOST_CONTENTION_LOG_CONTENDED_SAMPLES == 0)) || return 1
    host_contention_validate_thermal_alignment "$thermal_settle" \
      "$contention_settle" || return 1
    awk -F '\t' -v owner="$expected_owner_pgid" '
      NF != 6 || $3 != "loaded-idle" || $4 != owner { exit 1 }
    ' "$contention_settle" || return 1

    thermal_validate_measurement_log "$thermal_measurement" "$maximum_gap" \
      || return 1
    matched_validate_host_observation_log "$host_measurement" 3 1 \
      "$maximum_gap" || return 1
    matched_validate_calibration_alignment "$thermal_measurement" \
      "$host_measurement" || return 1
    host_contention_validate_measurement_log "$contention_measurement" \
      "$maximum_gap" || return 1
    host_contention_validate_thermal_alignment "$thermal_measurement" \
      "$contention_measurement" || return 1
    awk -F '\t' -v owner="$expected_owner_pgid" '
      NF != 6 || $4 != owner \
        || $3 !~ /^(measurement-start|measurement|measurement-end)$/ { exit 1 }
    ' "$contention_measurement"
}

matched_require_result_seal() {
    local child_dir=$1 summary_sha evidence_sha
    [[ -d "$child_dir" && ! -L "$child_dir" \
      && -f "$child_dir/summary.json" && ! -L "$child_dir/summary.json" \
      && -f "$child_dir/evidence.sha256" && ! -L "$child_dir/evidence.sha256" \
      && -f "$child_dir/result.sha256" && ! -L "$child_dir/result.sha256" ]] \
      || return 1
    qwen38_validate_evidence_manifest_paths "$child_dir/evidence.sha256" \
      || return 1
    [[ "$(awk 'END { print NR }' "$child_dir/result.sha256")" == 2 ]] \
      || return 1
    summary_sha=$(shasum -a 256 "$child_dir/summary.json" | awk '{print $1}') \
      || return 1
    evidence_sha=$(shasum -a 256 "$child_dir/evidence.sha256" \
      | awk '{print $1}') || return 1
    [[ "$(sed -n '1p' "$child_dir/result.sha256")" == \
        "$summary_sha  summary.json" \
      && "$(sed -n '2p' "$child_dir/result.sha256")" == \
        "$evidence_sha  evidence.sha256" ]] || return 1
    (cd "$child_dir" && shasum -a 256 -c evidence.sha256 >/dev/null \
      && shasum -a 256 -c result.sha256 >/dev/null)
}

matched_terminate_owned_child() {
    local child_pid=${1:-} deadline
    [[ -n "$child_pid" ]] || return 0
    [[ "$child_pid" =~ ^[1-9][0-9]*$ ]] || return 2
    if kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        deadline=$((SECONDS + 15))
        while kill -0 "$child_pid" 2>/dev/null && ((SECONDS < deadline)); do
            sleep 1
        done
    fi
    if kill -0 "$child_pid" 2>/dev/null; then
        kill -KILL "$child_pid" 2>/dev/null || true
    fi
    wait "$child_pid" 2>/dev/null || true
    ! kill -0 "$child_pid" 2>/dev/null
}

matched_validate_reopened_reference_child() {
    local child_dir=$1 expected_owner trial_dir trial_index=0
    local expected_engine
    matched_require_result_seal "$child_dir" || return 1
    expected_owner=$(jq -er '
      .calibration.host_contention as $host
      | select(.schema == 5 and .verdict == "pass"
          and .calibration.trial_logs == 24
          and $host.policy == "process-group-cpu-v2"
          and $host.maximum_foreign_cpu_percent == 100
          and $host.owner_scope == "release-gate-process-group"
          and ($host.owner_pgid | type == "number" and floor == . and . > 0)
          and $host.continuous == true)
      | $host.owner_pgid
    ' "$child_dir/summary.json") || return 1
    matched_validate_contention_preflight_log \
      "$child_dir/contention-preflight.tsv" 5 "$expected_owner" || return 1
    matched_require_evidence_manifest_entry "$child_dir" \
      "$child_dir/contention-preflight.tsv" || return 1
    [[ -d "$child_dir/trials" && ! -L "$child_dir/trials" ]] || return 1
    for expected_engine in hf2q reference reference hf2q; do
        trial_index=$((trial_index + 1))
        trial_dir="$child_dir/trials/trial-$trial_index-$expected_engine"
        matched_validate_reopened_trial_calibration "$trial_dir" \
          "$expected_owner" 120 5 "$child_dir" || return 1
    done
    [[ "$(find "$child_dir/trials" -mindepth 1 -maxdepth 1 -type d \
      -name 'trial-*' | wc -l | tr -d '[:space:]')" == 4 ]]
}

# Emit a sealed-shape stability receipt for the two observations of every
# engine/case and the two complete trial totals for every engine/group.
matched_measurement_stability_json() {
    local rows_file=$1
    local maximum_group_spread=$2
    local maximum_case_spread=$3

    jq -s --argjson maximum_group_spread "$maximum_group_spread" \
      --argjson maximum_case_spread "$maximum_case_spread" '
      def spread($values):
        ($values | min) as $minimum
        | ($values | max) as $maximum
        | if $minimum <= 0 then error("non-positive stability sample")
          else 100 * ($maximum - $minimum) / (($maximum + $minimum) / 2)
          end;
      def median($values):
        ($values | sort) as $sorted
        | ($sorted | length) as $count
        | if $count == 0 then error("empty median sample")
          elif ($count % 2) == 1 then $sorted[($count / 2 | floor)]
          else (($sorted[$count / 2 - 1] + $sorted[$count / 2]) / 2)
          end;
      . as $rows
      | ($rows | sort_by(.engine, .name) | group_by(.engine, .name)
        | map({engine:.[0].engine,name:.[0].name,samples:length,
            trials:(map(.trial) | sort),
            wall_spread_percent:spread(map(.wall_seconds)),
            decode_tps_spread_percent:spread(map(.internal_decode_tps)),
            completion_token_variants:(map(.completion_tokens) | unique | length)}))
        as $cases
      | ($rows | sort_by(.engine, .group, .trial)
        | group_by(.engine, .group, .trial)
        | map(. as $trial_rows
          | ($trial_rows | map(.completion_tokens) | add) as $tokens
          | ($trial_rows | map(.wall_seconds) | add) as $wall
          | ($trial_rows | map(.internal_decode_seconds) | add) as $decode
          | {engine:.[0].engine,group:.[0].group,trial:.[0].trial,
              samples:length,total_seconds:$wall,completion_tokens:$tokens,
              aggregate_decode_tps:($tokens / $decode),
              e2e_tps:($tokens / $wall)})
        | sort_by(.engine, .group)
        | group_by(.engine, .group)
        | map({engine:.[0].engine,group:.[0].group,
            samples_per_trial:(map(.samples)),trials:(map(.trial) | sort),
            trial_total_seconds:(map(.total_seconds)),
            trial_aggregate_decode_tps:(map(.aggregate_decode_tps)),
            trial_e2e_tps:(map(.e2e_tps)),
            median_trial_total_seconds:median(map(.total_seconds)),
            wall_spread_percent:spread(map(.total_seconds)),
            decode_tps_spread_percent:spread(map(.aggregate_decode_tps))})) as $groups
      | (["code","repeat"] | map(. as $group
          | ($groups[] | select(.engine == "hf2q" and .group == $group)) as $hf2q
          | ($groups[] | select(.engine == "reference" and .group == $group)) as $reference
          | {group:$group,
              hf2q_worst_total_seconds:($hf2q.trial_total_seconds | max),
              reference_best_total_seconds:($reference.trial_total_seconds | min),
              hf2q_worst_e2e_tps:($hf2q.trial_e2e_tps | min),
              reference_best_e2e_tps:($reference.trial_e2e_tps | max),
              wall_dominance:(($hf2q.trial_total_seconds | max)
                <= ($reference.trial_total_seconds | min)),
              e2e_tps_dominance:(($hf2q.trial_e2e_tps | min)
                >= ($reference.trial_e2e_tps | max))})) as $comparisons
      | {schema:1,total_samples:($rows | length),
          maximum_group_spread_percent:$maximum_group_spread,
          maximum_case_spread_percent:$maximum_case_spread,
          cases:$cases,groups:$groups,comparisons:$comparisons,
          observed_band_dominance:all($comparisons[];
            .wall_dominance and .e2e_tps_dominance),
          stable:(($rows | length) == 24
            and all($rows[];
              (.engine == "hf2q" or .engine == "reference")
              and (.name == "code-a" or .name == "code-b" or
                .name == "code-c" or .name == "repeat-a" or
                .name == "repeat-b" or .name == "repeat-c")
              and .group == (if (.name | startswith("code-"))
                then "code" else "repeat" end)
              and (.trial == (if .engine == "hf2q" then 1 else 2 end) or
                .trial == (if .engine == "hf2q" then 4 else 3 end))
              and (.wall_seconds | type == "number") and .wall_seconds > 0
              and (.internal_decode_tps | type == "number")
              and .internal_decode_tps > 0
              and (.internal_decode_seconds | type == "number")
              and .internal_decode_seconds > 0
              and (.completion_tokens | type == "number")
              and (.completion_tokens | floor) == .completion_tokens
              and .completion_tokens > 0)
            and ($cases | length) == 12
            and ($groups | length) == 4
            and all($cases[];
              .samples == 2
              and (.trials == (if .engine == "hf2q" then [1,4] else [2,3] end))
              and .completion_token_variants == 1
              and .wall_spread_percent <= ($maximum_case_spread + 1e-9)
              and .decode_tps_spread_percent <= ($maximum_case_spread + 1e-9))
            and all($groups[];
              .samples_per_trial == [3,3]
              and (.trials == (if .engine == "hf2q" then [1,4] else [2,3] end))
              and .wall_spread_percent <= ($maximum_group_spread + 1e-9)
              and .decode_tps_spread_percent <= ($maximum_group_spread + 1e-9)))}
    ' "$rows_file"
}

matched_publish_result() {
    local summary_tmp=$1
    local summary_final=$2
    local evidence_manifest=$3
    local result_final=$4
    local output_dir result_tmp summary_sha evidence_sha

    output_dir=$(dirname "$summary_final")
    result_tmp="$result_final.tmp"
    [[ "$(dirname "$evidence_manifest")" == "$output_dir"
      && "$(dirname "$result_final")" == "$output_dir"
      && "$(basename "$summary_final")" == summary.json
      && "$(basename "$evidence_manifest")" == evidence.sha256
      && "$(basename "$result_final")" == result.sha256
      && ! -e "$summary_final" && ! -e "$result_final" ]] || return 1

    summary_sha=$(shasum -a 256 "$summary_tmp" | awk '{print $1}') || return 1
    evidence_sha=$(shasum -a 256 "$evidence_manifest" | awk '{print $1}') \
      || return 1
    (cd "$output_dir" && shasum -a 256 -c evidence.sha256 >/dev/null) \
      || return 1
    printf '%s  summary.json\n%s  evidence.sha256\n' \
      "$summary_sha" "$evidence_sha" >"$result_tmp" || return 1
    [[ "$(sed -n '1p' "$result_tmp")" == "$summary_sha  summary.json"
      && "$(sed -n '2p' "$result_tmp")" == "$evidence_sha  evidence.sha256"
      && "$(awk 'END { print NR }' "$result_tmp")" == 2 ]] || return 1

    # Publish the seal first. The passing summary is the final atomic write, so
    # no failed sealing step can leave a visible pass claim behind.
    mv "$result_tmp" "$result_final" || return 1
    mv "$summary_tmp" "$summary_final"
}
