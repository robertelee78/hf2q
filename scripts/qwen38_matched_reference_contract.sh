#!/usr/bin/env bash

# Pure, model-free predicates shared by the matched Qwen3.8 runner and its
# hosted contract test. The caller owns `set -euo pipefail`.

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
      and (.choices[0].finish_reason | type == "string" and length > 0)
      and (.usage.prompt_tokens | type == "number" and floor == . and . > 0)
      and (.usage.completion_tokens | type == "number" and floor == . and . > 0)
    ' "$response" >/dev/null
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
    local samples duration gaps invalid starts ends

    stats=$(awk -F '\t' -v maximum="$maximum_gap" '
      BEGIN { invalid = 0; gaps = 0 }
      {
        if (NF != 4 || $1 !~ /^[0-9]+$/ || $2 != "ac" || $3 != "quiet" ||
          length($4) == 0) {
          invalid++
          next
        }
        samples++
        if (samples == 1) first = $1
        if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
        previous = $1
        last = $1
        phases[$4]++
      }
      END {
        duration = samples > 0 ? last - first : -1
        printf "%d\t%d\t%d\t%d\t%d\t%d\n", samples, duration, gaps,
          invalid, phases["measurement-start"], phases["measurement-end"]
      }
    ' "$host_log") || return 1
    IFS=$'\t' read -r samples duration gaps invalid starts ends <<<"$stats"
    ((samples >= minimum_samples && duration >= required_duration \
      && gaps == 0 && invalid == 0)) || return 1
    if ((minimum_samples >= 3)); then
        ((starts == 1 && ends == 1)) || return 1
        awk -F '\t' '$4 == "measurement" { found = 1 }
          END { exit !found }' "$host_log"
    fi
}

# Read `ps -axo pid=,command=` rows from stdin and report Python processes
# whose command line identifies model work. Keep this parser in the sourced
# contract so the macOS awk implementation used by production is exercised by
# the hosted-safe behavioral test as well.
matched_find_scripted_model_work() {
    local allowed_pid=$1
    awk -v allowed="$allowed_pid" '
      {
        pid = $1
        $1 = ""
        sub(/^[[:space:]]+/, "", $0)
        command = tolower($0)
        if (pid != allowed && command ~ /(^|\/)python(3([.][0-9]+)?)?([[:space:]]|$)/ && command ~ /(mlx|torch|transformers|teacher|model[-_ ]?gen|inference|vllm)/) {
          print pid ":python-model-work"
        }
      }
    '
}

matched_validate_calibration_alignment() {
    local thermal_log=$1
    local host_log=$2
    cmp -s \
      <(awk -F '\t' 'NF == 3 { print $1 "\t" $3 }' "$thermal_log") \
      <(awk -F '\t' 'NF == 4 { print $1 "\t" $4 }' "$host_log")
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
