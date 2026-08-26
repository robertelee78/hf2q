#!/usr/bin/env bash

# Model-free predicates for the matched physical Qwen3.8 ABBA runners.
# Callers own `set -euo pipefail` and source the artifact, physical-width, and
# matched-reference contracts before using this file.

matched_physical_require_clean_exact_source() {
    local source_dir=$1 expected=$2 label=$3 actual status
    actual=$(git -C "$source_dir" rev-parse HEAD) || return 1
    [[ "$actual" == "$expected" ]] || {
        echo "$label HEAD mismatch: expected=$expected actual=$actual" >&2
        return 1
    }
    status=$(git -C "$source_dir" status --porcelain) || return 1
    [[ -z "$status" ]] || {
        echo "$label source must be clean" >&2
        printf '%s\n' "$status" >&2
        return 1
    }
}

matched_physical_parse_sse_stream() {
    local started_at=$1
    local stream_path=$2
    local receipt_path=$3
    local expected_content=$4

    perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -MJSON::PP -e '
      use strict;
      use warnings;
      my ($started, $stream_path, $receipt_path, $expected) = @ARGV;
      open my $stream, ">", $stream_path or die "open stream: $!";
      my ($first_semantic_ms, $content, $done_count, $event_count) =
        (undef, "", 0, 0);
      my $done_seen = 0;
      my ($role_seen, $finish_reason, $finish_count) = (0, undef, 0);
      my ($prompt_tokens, $completion_tokens);
      while (my $line = <STDIN>) {
        print {$stream} $line or die "write stream: $!";
        next unless $line =~ /^data:\s?(.*?)[\r\n]*$/;
        my $payload = $1;
        die "SSE data followed terminal DONE" if $done_seen;
        if ($payload eq "[DONE]") {
          $done_count++;
          $done_seen = 1;
          next;
        }
        my $event = eval { decode_json($payload) };
        die "malformed SSE JSON: $@" if $@;
        $event_count++;
        if (ref($event->{usage}) eq "HASH") {
          $prompt_tokens = $event->{usage}{prompt_tokens}
            if defined($event->{usage}{prompt_tokens});
          $completion_tokens = $event->{usage}{completion_tokens}
            if defined($event->{usage}{completion_tokens});
        }
        my $choices = $event->{choices};
        die "SSE choices must be an array" unless ref($choices) eq "ARRAY";
        if (@{$choices} == 0) {
          die "choice-less SSE event omitted usage" unless ref($event->{usage}) eq "HASH";
          next;
        }
        die "SSE choice cardinality mismatch" unless @{$choices} == 1;
        my $choice = $choices->[0];
        my $delta = $choice->{delta};
        if (ref($delta) eq "HASH") {
          $role_seen = 1 if defined($delta->{role}) && $delta->{role} eq "assistant";
          my $semantic = 0;
          if (defined($delta->{reasoning_content}) && !ref($delta->{reasoning_content})
              && length($delta->{reasoning_content}) > 0) {
            die "transcription stream emitted reasoning content";
          }
          if (defined($delta->{content}) && !ref($delta->{content})
              && length($delta->{content}) > 0) {
            $semantic = 1;
            $content .= $delta->{content};
          }
          if (ref($delta->{tool_calls}) eq "ARRAY" && @{$delta->{tool_calls}} > 0) {
            die "transcription stream emitted a tool call";
          }
          if (defined($delta->{refusal}) && !ref($delta->{refusal})
              && length($delta->{refusal}) > 0) {
            die "transcription stream emitted a refusal";
          }
          if ($semantic && !defined($first_semantic_ms)) {
            $first_semantic_ms =
              (clock_gettime(CLOCK_MONOTONIC) - $started) * 1000.0;
          }
        }
        if (defined($choice->{finish_reason})) {
          $finish_reason = $choice->{finish_reason};
          $finish_count++;
        }
      }
      close $stream or die "close stream: $!";
      my $ended = clock_gettime(CLOCK_MONOTONIC);
      die "stream omitted semantic output" unless defined($first_semantic_ms);
      die "stream did not terminate exactly once" unless $done_count == 1;
      die "stream omitted assistant role" unless $role_seen;
      die "stream finish cardinality mismatch" unless $finish_count == 1;
      die "streamed content mismatch" unless $content eq $expected;
      die "stream omitted prompt usage" unless defined($prompt_tokens)
        && $prompt_tokens =~ /^\d+$/ && $prompt_tokens > 0;
      die "stream omitted completion usage" unless defined($completion_tokens)
        && $completion_tokens =~ /^\d+$/ && $completion_tokens > 0;
      open my $receipt, ">", $receipt_path or die "open receipt: $!";
      print {$receipt} JSON::PP->new->canonical->encode({
        schema => 1,
        role => "assistant",
        first_semantic_ms => 0.0 + $first_semantic_ms,
        wall_seconds => 0.0 + ($ended - $started),
        content => $content,
        finish_reason => $finish_reason,
        prompt_tokens => 0 + $prompt_tokens,
        completion_tokens => 0 + $completion_tokens,
        done_count => $done_count,
        event_count => $event_count,
      }), "\n" or die "write receipt: $!";
      close $receipt or die "close receipt: $!";
    ' "$started_at" "$stream_path" "$receipt_path" "$expected_content"
}

matched_physical_validate_repeat_scalar() {
    local stream_receipt=$1
    local scalar_response=$2
    local concurrent scalar

    jq -e '
      ((.choices[0].message.tool_calls // null) == null)
      and ((.choices[0].message.reasoning_content // null) == null)
      and ((.choices[0].message.refusal // null) == null)
    ' "$scalar_response" >/dev/null || return 1
    concurrent=$(jq -Sce '{
      message:{role:.role,content:.content},finish_reason,
      prompt_tokens,completion_tokens
    }' "$stream_receipt") || return 1
    scalar=$(jq -Sce '{
      message:{role:.choices[0].message.role,content:.choices[0].message.content},
      finish_reason:.choices[0].finish_reason,
      prompt_tokens:.usage.prompt_tokens,
      completion_tokens:.usage.completion_tokens
    }' "$scalar_response") || return 1
    [[ "$concurrent" == "$scalar" ]] || {
        echo "streamed response diverged from scalar replay" >&2
        echo "concurrent=$concurrent" >&2
        echo "scalar=$scalar" >&2
        return 1
    }
}

# The exact-repeat fixture is a common byte sequence, whereas OpenAI usage
# counters are engine-owned accounting (for example, one engine may include
# terminal EOS and another may not). Derive the comparable work count once
# from hf2q's GGUF tokenizer with special-token insertion disabled. The
# receipt binds the count and token stream to the exact artifact, candidate
# binary, fixture bytes, and clean candidate source identity. It performs only
# GGUF metadata/tokenizer work; `generate` exits before model allocation.
matched_physical_record_semantic_repeat_tokens() {
    local hf2q_bin=$1 model_path=$2 model_sha256=$3 model_snapshot=$4
    local hf2q_sha256=$5 hf2q_commit=$6 expected_path=$7 output=$8
    local raw_template ids output_text count ids_sha expected_sha expected_bytes

    [[ -x "$hf2q_bin" && -f "$model_path" && -r "$model_path" \
      && -f "$expected_path" && -r "$expected_path" ]] || return 1
    [[ "$model_sha256" =~ ^[0-9a-f]{64}$ && "$hf2q_sha256" =~ ^[0-9a-f]{64}$ \
      && "$hf2q_commit" =~ ^[0-9a-f]{40}$ && -n "$model_snapshot" ]] || return 1
    raw_template="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen35_raw_passthrough.jinja"
    [[ -f "$raw_template" && -r "$raw_template" ]] || return 1

    output_text=$(HF2Q_DEBUG_TOKENIZE_NO_SPECIAL_TOKENS=1 "$hf2q_bin" generate \
      --model "$model_path" --chat-template-file "$raw_template" \
      --prompt-file "$expected_path" --max-tokens 1 --temperature 0) || return 1
    [[ "$(printf '%s\n' "$output_text" | awk '/^TOKENIZE_DEBUG_IDS:/{n++} END {print n+0}')" == 1 ]] \
      || return 1
    ids=$(printf '%s\n' "$output_text" \
      | awk '/^TOKENIZE_DEBUG_IDS:/{sub(/^TOKENIZE_DEBUG_IDS:[[:space:]]*/, ""); print}')
    [[ -n "$ids" ]] || return 1
    count=$(printf '%s\n' "$ids" | awk '
      { for (i = 1; i <= NF; i++) if ($i !~ /^[0-9]+$/) exit 1; count += NF }
      END { if (count < 1) exit 1; print count }
    ') || return 1
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || return 1
    ids_sha=$(printf '%s\n' "$ids" | shasum -a 256 | awk '{print $1}') || return 1
    expected_sha=$(shasum -a 256 "$expected_path" | awk '{print $1}') || return 1
    expected_bytes=$(wc -c <"$expected_path" | tr -d '[:space:]') || return 1
    [[ "$ids_sha" =~ ^[0-9a-f]{64}$ && "$expected_sha" =~ ^[0-9a-f]{64}$ \
      && "$expected_bytes" =~ ^[1-9][0-9]*$ ]] || return 1

    jq -n --arg model_path "$model_path" --arg model_sha256 "$model_sha256" \
      --arg model_snapshot "$model_snapshot" --arg hf2q_sha256 "$hf2q_sha256" \
      --arg hf2q_commit "$hf2q_commit" --arg expected_sha256 "$expected_sha" \
      --argjson expected_bytes "$expected_bytes" --arg ids_sha256 "$ids_sha" \
      --argjson semantic_completion_tokens "$count" '{
        schema:1,method:"hf2q-gguf-no-special-tokens-v1",
        model:{path:$model_path,sha256:$model_sha256,file_snapshot:$model_snapshot},
        tokenizer:{binary_sha256:$hf2q_sha256,source_commit:$hf2q_commit,
          token_ids_sha256:$ids_sha256},
        expected:{sha256:$expected_sha256,bytes:$expected_bytes},
        semantic_completion_tokens:$semantic_completion_tokens
      }' >"$output"
}

matched_physical_validate_semantic_repeat_tokens() {
    local receipt=$1 model_path=$2 model_sha256=$3 model_snapshot=$4
    local hf2q_sha256=$5 hf2q_commit=$6 expected_path=$7
    local expected_sha expected_bytes

    [[ -f "$receipt" && -r "$receipt" && -f "$expected_path" && -r "$expected_path" ]] \
      || return 1
    expected_sha=$(shasum -a 256 "$expected_path" | awk '{print $1}') || return 1
    expected_bytes=$(wc -c <"$expected_path" | tr -d '[:space:]') || return 1
    jq -e --arg model_path "$model_path" --arg model_sha256 "$model_sha256" \
      --arg model_snapshot "$model_snapshot" --arg hf2q_sha256 "$hf2q_sha256" \
      --arg hf2q_commit "$hf2q_commit" --arg expected_sha256 "$expected_sha" \
      --argjson expected_bytes "$expected_bytes" '
        .schema == 1
        and .method == "hf2q-gguf-no-special-tokens-v1"
        and .model == {path:$model_path,sha256:$model_sha256,file_snapshot:$model_snapshot}
        and .tokenizer.binary_sha256 == $hf2q_sha256
        and .tokenizer.source_commit == $hf2q_commit
        and (.tokenizer.token_ids_sha256 | type == "string"
          and test("^[0-9a-f]{64}$"))
        and .expected == {sha256:$expected_sha256,bytes:$expected_bytes}
        and (.semantic_completion_tokens | type == "number"
          and . > 0 and . == floor)
      ' "$receipt" >/dev/null
}

matched_physical_validate_launch_skew() {
    local clients_path=$1
    local maximum_seconds=$2
    jq -e --argjson maximum "$maximum_seconds" '
      length > 0
      and all(.[]; (.started_at | type == "number") and .started_at > 0)
      and ((map(.started_at) | max) - (map(.started_at) | min) <= $maximum)
    ' "$clients_path" >/dev/null
}

matched_physical_validate_client_overlap() {
    local clients_path=$1
    jq -e '
      length > 0
      and all(.[];
        (.started_at | type == "number") and .started_at > 0
        and (.ended_at | type == "number") and .ended_at > .started_at)
      and ((map(.started_at) | max) < (map(.ended_at) | min))
    ' "$clients_path" >/dev/null
}

matched_physical_owned_pid_live() {
    local owned_pid=$1 state
    kill -0 "$owned_pid" 2>/dev/null || return 1
    state=$(ps -o stat= -p "$owned_pid" 2>/dev/null | awk 'NR == 1 {print $1}')
    [[ -n "$state" && "$state" != Z* ]]
}

# Cleanup is allowed to report a failed monitor, but never before it has
# stopped the owned server and proved that the listener is gone.
matched_physical_stop_owned_server() {
    local owned_server_pid=$1 port=$2 owned_monitor_pid=$3 monitor_stop_file=$4
    local waited=0 cleanup_rc=0
    local int_grace=${MATCHED_PHYSICAL_SERVER_INT_GRACE_SECONDS:-60}
    local term_grace=${MATCHED_PHYSICAL_SERVER_TERM_GRACE_SECONDS:-30}
    [[ "$int_grace" =~ ^[0-9]+$ && "$term_grace" =~ ^[0-9]+$ ]] || return 1
    if [[ -n "$monitor_stop_file" ]] && ! : >"$monitor_stop_file"; then
        cleanup_rc=1
        if [[ -n "$owned_monitor_pid" ]]; then
            kill -TERM "$owned_monitor_pid" 2>/dev/null || true
        fi
    fi
    if [[ -n "$owned_monitor_pid" ]]; then
        wait "$owned_monitor_pid" 2>/dev/null || cleanup_rc=1
    fi
    if [[ -n "$owned_server_pid" ]] \
      && matched_physical_owned_pid_live "$owned_server_pid"; then
        kill -INT "$owned_server_pid" 2>/dev/null || true
        while matched_physical_owned_pid_live "$owned_server_pid" \
          && ((waited < int_grace)); do
            sleep 1
            waited=$((waited + 1))
        done
    fi
    if [[ -n "$owned_server_pid" ]] \
      && matched_physical_owned_pid_live "$owned_server_pid"; then
        kill -TERM "$owned_server_pid" 2>/dev/null || true
        waited=0
        while matched_physical_owned_pid_live "$owned_server_pid" \
          && ((waited < term_grace)); do
            sleep 1
            waited=$((waited + 1))
        done
    fi
    if [[ -n "$owned_server_pid" ]] \
      && matched_physical_owned_pid_live "$owned_server_pid"; then
        kill -KILL "$owned_server_pid" 2>/dev/null || true
    fi
    if [[ -n "$owned_server_pid" ]]; then
        wait "$owned_server_pid" 2>/dev/null || true
    fi
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null \
      | sed -n '2p' | rg -q .; then
        echo "owned server listener remained after cleanup: $port" >&2
        cleanup_rc=1
    fi
    return "$cleanup_rc"
}

matched_physical_terminate_owned_child() {
    local child_pid=$1 signal=${2:-TERM} waited=0
    [[ -n "$child_pid" ]] || return 0
    if matched_physical_owned_pid_live "$child_pid"; then
        kill -"$signal" "$child_pid" 2>/dev/null || true
        while matched_physical_owned_pid_live "$child_pid" \
          && ((waited < 60)); do
            sleep 1
            waited=$((waited + 1))
        done
    fi
    if matched_physical_owned_pid_live "$child_pid"; then
        echo "matched matrix child ignored bounded shutdown; killing owned child $child_pid" >&2
        kill -KILL "$child_pid" 2>/dev/null || true
    fi
    wait "$child_pid" 2>/dev/null || true
}

matched_physical_validate_processing_peak() {
    local width=$1
    local samples_path=$2
    local samples peak invalid

    [[ "$width" =~ ^(1|2|4|8|16)$ ]] || return 1
    [[ -s "$samples_path" ]] || return 1
    read -r samples peak invalid < <(awk -v width="$width" '
      NF != 2 || $1 !~ /^[0-9]+([.][0-9]+)?$/ || $2 !~ /^[0-9]+$/ {
        invalid++
        next
      }
      { samples++; if ($2 > peak) peak = $2; if ($2 > width) invalid++ }
      END { print samples + 0, peak + 0, invalid + 0 }
    ' "$samples_path")
    ((samples >= 1 && peak == width && invalid == 0)) || {
        echo "processing peak mismatch: width=$width samples=$samples peak=$peak invalid=$invalid" >&2
        return 1
    }
}

matched_physical_metric_u64() {
    local path=$1 name=$2
    awk -v name="$name" '
      $1 == name {
        if (++matches != 1 || NF != 2 || $2 !~ /^[0-9]+$/) exit 2
        value = $2
      }
      END { if (matches != 1) exit 3; print value }
    ' "$path"
}

matched_physical_hf2q_sum_u64() {
    local path=$1 name=$2
    awk -v name="$name" '
      $1 ~ ("^" name "\\{proposer=\\\"(history_lookup|mtp)\\\"\\}$") {
        if (NF != 2 || $2 !~ /^[0-9]+$/) exit 2
        sum += $2
        matches++
      }
      END { if (matches != 2) exit 3; print sum + 0 }
    ' "$path"
}

matched_physical_hf2q_sum_f64() {
    local path=$1 name=$2
    awk -v name="$name" '
      function numeric(value) {
        return value ~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/
      }
      $1 ~ ("^" name "\\{proposer=\\\"(history_lookup|mtp)\\\"\\}$") {
        if (NF != 2 || !numeric($2)) exit 2
        sum += $2
        matches++
      }
      END { if (matches != 2) exit 3; printf "%.9f\n", sum + 0 }
    ' "$path"
}

# hf2q's adaptive shipping policy may stop proposing when measured end-to-end
# cost loses to ordinary decode. Each timed wave therefore proves policy
# execution by either accepted draft tokens or a cost-disable backed by proposal
# and timing deltas. The fixed-K3 reference route has no analogous cost gate and
# must draft and accept within every timed wave.
matched_physical_validate_wave_speculation() {
    local engine=$1 trial=$2 group=$3 before=$4 after=$5 output=$6
    local proposals_before proposals_after drafted_before drafted_after
    local accepted_before accepted_after cost_before=0 cost_after=0
    local round_before=0 round_after=0 ordinary_before=0 ordinary_after=0
    local proposals drafted accepted cost_disabled round_seconds ordinary_seconds
    local proof_mode disable_reason=''

    [[ "$engine" == hf2q || "$engine" == reference ]] || return 1
    [[ "$trial" =~ ^[1-4]$ ]] || return 1
    [[ "$group" == code || "$group" == repeat ]] || return 1
    if [[ "$engine" == hf2q ]]; then
        proposals_before=$(matched_physical_hf2q_sum_u64 \
          "$before" hf2q_qwen_speculation_proposals_total) || return 1
        proposals_after=$(matched_physical_hf2q_sum_u64 \
          "$after" hf2q_qwen_speculation_proposals_total) || return 1
        drafted_before=$(matched_physical_hf2q_sum_u64 \
          "$before" hf2q_qwen_speculation_drafted_tokens_total) || return 1
        drafted_after=$(matched_physical_hf2q_sum_u64 \
          "$after" hf2q_qwen_speculation_drafted_tokens_total) || return 1
        accepted_before=$(matched_physical_hf2q_sum_u64 \
          "$before" hf2q_qwen_speculation_accepted_tokens_total) || return 1
        accepted_after=$(matched_physical_hf2q_sum_u64 \
          "$after" hf2q_qwen_speculation_accepted_tokens_total) || return 1
        cost_before=$(matched_physical_hf2q_sum_u64 \
          "$before" hf2q_qwen_speculation_cost_disabled_total) || return 1
        cost_after=$(matched_physical_hf2q_sum_u64 \
          "$after" hf2q_qwen_speculation_cost_disabled_total) || return 1
        round_before=$(matched_physical_hf2q_sum_f64 \
          "$before" hf2q_qwen_speculation_round_seconds_total) || return 1
        round_after=$(matched_physical_hf2q_sum_f64 \
          "$after" hf2q_qwen_speculation_round_seconds_total) || return 1
        ordinary_before=$(matched_physical_hf2q_sum_f64 \
          "$before" hf2q_qwen_speculation_equivalent_ordinary_seconds_total) \
          || return 1
        ordinary_after=$(matched_physical_hf2q_sum_f64 \
          "$after" hf2q_qwen_speculation_equivalent_ordinary_seconds_total) \
          || return 1
    else
        proposals_before=$(matched_physical_metric_u64 \
          "$before" llamacpp:spec_decode_num_drafts_total) || return 1
        proposals_after=$(matched_physical_metric_u64 \
          "$after" llamacpp:spec_decode_num_drafts_total) || return 1
        drafted_before=$(matched_physical_metric_u64 \
          "$before" llamacpp:spec_decode_num_draft_tokens_total) || return 1
        drafted_after=$(matched_physical_metric_u64 \
          "$after" llamacpp:spec_decode_num_draft_tokens_total) || return 1
        accepted_before=$(matched_physical_metric_u64 \
          "$before" llamacpp:spec_decode_num_accepted_tokens_total) || return 1
        accepted_after=$(matched_physical_metric_u64 \
          "$after" llamacpp:spec_decode_num_accepted_tokens_total) || return 1
    fi
    ((proposals_after >= proposals_before && drafted_after >= drafted_before \
      && accepted_after >= accepted_before && cost_after >= cost_before)) || return 1
    proposals=$((proposals_after - proposals_before))
    drafted=$((drafted_after - drafted_before))
    accepted=$((accepted_after - accepted_before))
    cost_disabled=$((cost_after - cost_before))
    ((accepted <= drafted)) || return 1
    round_seconds=$(awk -v after="$round_after" -v before="$round_before" \
      'BEGIN { printf "%.9f", after-before }')
    ordinary_seconds=$(awk -v after="$ordinary_after" -v before="$ordinary_before" \
      'BEGIN { printf "%.9f", after-before }')
    awk -v round="$round_seconds" -v ordinary="$ordinary_seconds" \
      'BEGIN { exit !(round >= 0 && ordinary >= 0) }' || return 1

    if ((proposals > 0 && drafted > 0 && accepted > 0)); then
        proof_mode='accepted-proposals'
    elif [[ "$engine" == hf2q ]] && ((proposals > 0 && drafted > 0 \
      && cost_disabled > 0)) && awk -v round="$round_seconds" \
      -v ordinary="$ordinary_seconds" \
      'BEGIN { exit !(round > 0 && ordinary > 0) }'; then
        proof_mode='measured-cost-disabled'
        disable_reason='measured_cost_unprofitable'
    else
        echo "$engine trial $trial did not prove shipping speculation policy execution" >&2
        return 1
    fi
    [[ "$engine" == hf2q || "$proof_mode" == accepted-proposals ]] || return 1

    local policy
    if [[ "$engine" == hf2q ]]; then
        policy=$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY
    else
        policy=$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY
    fi
    jq -n --arg engine "$engine" --argjson trial "$trial" --arg group "$group" \
      --arg policy "$policy" \
      --arg proof_mode "$proof_mode" --arg disable_reason "$disable_reason" \
      --argjson proposals "$proposals" --argjson drafted "$drafted" \
      --argjson accepted "$accepted" --argjson cost_disabled "$cost_disabled" \
      --argjson round_seconds "$round_seconds" \
      --argjson ordinary_seconds "$ordinary_seconds" '{
        schema:1,engine:$engine,trial:$trial,group:$group,policy:$policy,
        proof_pass:true,proof_mode:$proof_mode,
        disable_reason:(if $disable_reason == "" then null else $disable_reason end),
        proposals:$proposals,drafted_tokens:$drafted,accepted_tokens:$accepted,
        cost_disabled_generations:$cost_disabled,
        measured_round_seconds:$round_seconds,
        equivalent_ordinary_seconds:$ordinary_seconds
      }' >"$output"
}

matched_physical_group_result_json() {
    local rows_path=$1
    local width=$2
    local group=$3
    local maximum_group_spread=$4
    local maximum_case_spread=$5

    jq -s --argjson width "$width" --arg group "$group" \
      --argjson max_group_spread "$maximum_group_spread" \
      --argjson max_case_spread "$maximum_case_spread" '
      def spread($values):
        ($values | min) as $lo | ($values | max) as $hi
        | if $lo <= 0 then error("non-positive stability sample")
          else 100 * ($hi - $lo) / (($hi + $lo) / 2) end;
      def median($values):
        ($values | sort) as $v | ($v | length) as $n
        | if $n == 0 then error("empty median")
          elif ($n % 2) == 1 then $v[($n / 2 | floor)]
          else (($v[$n / 2 - 1] + $v[$n / 2]) / 2) end;
      def p95($values):
        ($values | sort) as $v | ($v | length) as $n
        | if $n == 0 then error("empty p95")
          else $v[((($n * 95 + 99) / 100 | floor) - 1)] end;
      def close($a;$b):
        (($a - $b) | abs) <= (0.000001 * ([1, ($b | abs)] | max));
      [.[] | select(.width == $width and .group == $group)] as $rows
      | ($rows | all(.[]; . as $wave
          | .schema == 1
          and (.clients | length) == $width
          and ([.clients[].lane] | sort) == [range(1; $width + 1)]
          and .api_concurrency_proven == true
          and all(.clients[];
            (.started_at | type == "number") and .started_at > 0
            and (.ended_at | type == "number") and .ended_at > .started_at
            and (.wall_seconds | type == "number") and .wall_seconds > 0
            and close(.wall_seconds; (.ended_at - .started_at))
            and (.prompt_tokens | type == "number") and .prompt_tokens > 0
            and .prompt_tokens == (.prompt_tokens | floor)
            and (.completion_tokens | type == "number") and .completion_tokens > 0
            and .completion_tokens == (.completion_tokens | floor)
            and (if .first_semantic_ms == null then true
              else (.first_semantic_ms | type == "number")
                and .first_semantic_ms > 0
                and .first_semantic_ms <= (.wall_seconds * 1000) end)
            and .scalar_parity == true)
          and (.wave_started_at | type == "number") and .wave_started_at > 0
          and (.wave_ended_at | type == "number")
          and .wave_ended_at > .wave_started_at
          and (.wave_wall_seconds | type == "number") and .wave_wall_seconds > 0
          and close(.wave_wall_seconds; (.wave_ended_at - .wave_started_at))
          and all(.clients[]; .started_at >= $wave.wave_started_at)
          and all(.clients[]; .ended_at <= $wave.wave_ended_at)
          and .wave_wall_seconds >= ([.clients[].wall_seconds] | max)
          and (.total_completion_tokens | type == "number")
          and .total_completion_tokens > 0
          and .total_completion_tokens == (.total_completion_tokens | floor)
          and .total_completion_tokens == ([.clients[].completion_tokens] | add)
          and (.diagnostics.api_completion_tokens_per_second | type == "number")
          and .diagnostics.api_completion_tokens_per_second > 0
          and close(.diagnostics.api_completion_tokens_per_second;
            (.total_completion_tokens / .wave_wall_seconds))
          and (.comparison_work_units | type == "number")
          and .comparison_work_units > 0
          and (.comparison_units_per_second | type == "number")
          and .comparison_units_per_second > 0
          and close(.comparison_units_per_second;
            (.comparison_work_units / .wave_wall_seconds))
          and (if $group == "repeat" then
            .comparison_unit == "canonical-semantic-output-token"
            and all(.clients[];
              (.semantic_completion_tokens | type == "number")
              and .semantic_completion_tokens > 0
              and .semantic_completion_tokens == (.semantic_completion_tokens | floor)
              and (.semantic_tokenization_sha256 | type == "string"
                and test("^[0-9a-f]{64}$")))
            and (.total_semantic_completion_tokens | type == "number")
            and .total_semantic_completion_tokens > 0
            and .total_semantic_completion_tokens
              == ([.clients[].semantic_completion_tokens] | add)
            and .comparison_work_units == .total_semantic_completion_tokens
          else
            .comparison_unit == "evaluator-valid-code-request"
            and .comparison_work_units == $width
          end))) as $measurement_consistent
      | ($rows | map(. + {
          wave_p95_wall:p95([.clients[].wall_seconds]),
          wave_p95_ttft:(if all(.clients[];
              (.first_semantic_ms | type == "number") and .first_semantic_ms > 0)
            then p95([.clients[].first_semantic_ms]) else null end)})) as $waves
      | ($waves | sort_by(.engine, .trial) | group_by(.engine)
        | map({engine:.[0].engine,samples:length,trials:(map(.trial) | sort),
            comparison_rates:(map(.comparison_units_per_second)),
            api_completion_token_rates:
              (map(.diagnostics.api_completion_tokens_per_second)),
            wave_p95_wall:(map(.wave_p95_wall)),
            wave_p95_ttft:(map(.wave_p95_ttft) | map(select(. != null))),
            comparison_rate_spread_percent:spread(map(.comparison_units_per_second)),
            p95_spread_percent:spread(map(.wave_p95_wall)),
            ttft_spread_percent:(if all(.[]; .wave_p95_ttft != null)
              then spread(map(.wave_p95_ttft)) else null end),
            median_comparison_rate:median(map(.comparison_units_per_second)),
            median_api_completion_token_rate:
              median(map(.diagnostics.api_completion_tokens_per_second)),
            median_p95_wall:median(map(.wave_p95_wall)),
            median_p95_ttft:(if all(.[]; .wave_p95_ttft != null)
              then median(map(.wave_p95_ttft)) else null end)})) as $engines
      | ($rows | map(.engine as $engine | .trial as $trial | .clients[]
          | . + {engine:$engine,trial:$trial}) | sort_by(.engine, .lane)
        | group_by(.engine, .lane)
        | map({engine:.[0].engine,lane:.[0].lane,samples:length,
            wall_spread_percent:spread(map(.wall_seconds)),
            ttft_spread_percent:(if all(.[];
                (.first_semantic_ms | type == "number") and .first_semantic_ms > 0)
              then spread(map(.first_semantic_ms)) else null end),
            prompt_token_variants:(map(.prompt_tokens) | unique | length),
            completion_token_variants:(map(.completion_tokens) | unique | length),
            scalar_parity:all(.[]; .scalar_parity == true)})) as $lanes
      | ($rows | map(.engine as $engine | .trial as $trial | .clients[]
          | . + {engine:$engine,trial:$trial}) | sort_by(.lane)
        | group_by(.lane)
        | map({lane:.[0].lane,samples:length,
            engines:(map(.engine) | sort),trials:(map(.trial) | sort),
            prompt_token_variants:(map(.prompt_tokens) | unique | length),
            completion_token_variants:(map(.completion_tokens) | unique | length),
            semantic_completion_token_variants:(if $group == "repeat"
              then (map(.semantic_completion_tokens) | unique | length) else null end),
            semantic_tokenization_sha256_variants:(if $group == "repeat"
              then (map(.semantic_tokenization_sha256) | unique | length) else null end),
            semantic_tokenization_sha256:(if $group == "repeat" then
              (map(.semantic_tokenization_sha256) | unique
                | if length == 1 then .[0] else null end)
              else null end)}))
        as $cross_engine_lanes
      | ($group == "repeat") as $requires_latency
      | (($lanes | length) == (2 * $width)
          and all($lanes[]; .samples == 2 and .prompt_token_variants == 1
            and .completion_token_variants == 1)
          and ($cross_engine_lanes | length) == $width
          and all($cross_engine_lanes[];
            .samples == 4
            and .engines == ["hf2q","hf2q","reference","reference"]
            and .trials == [1,2,3,4]
            and .prompt_token_variants == 1
            and (if $requires_latency then
              .semantic_completion_token_variants == 1
              and .semantic_tokenization_sha256_variants == 1
            else true end))) as $token_accounting_pass
      | ($engines[] | select(.engine == "hf2q")) as $hf2q
      | ($engines[] | select(.engine == "reference")) as $reference
      | {width:$width,group:$group,
          quality_pass:all($rows[]; .quality_pass == true),
          measurement_consistent:$measurement_consistent,
          api_concurrency_pass:all($rows[]; .api_concurrency_proven == true),
          token_accounting:{pass:$token_accounting_pass,
            cross_engine_prompt_equality_required:true,
            raw_api_completion_equality_required_within_engine:true,
            cross_engine_completion_equality_required:false,
            cross_engine_semantic_completion_equality_required:$requires_latency,
            semantic_tokenization_sha256:(if $requires_latency then
              ($cross_engine_lanes[0].semantic_tokenization_sha256) else null end),
            lanes:$cross_engine_lanes},
          stability:{
            stable:(($rows | length) == 4
              and $measurement_consistent and $token_accounting_pass
              and ($engines | length) == 2
              and all($engines[];
                .samples == 2
                and (.trials == (if .engine == "hf2q" then [1,4] else [2,3] end))
                and .comparison_rate_spread_percent <= ($max_group_spread + 1e-9)
                and (if $requires_latency then
                    .p95_spread_percent <= ($max_group_spread + 1e-9)
                    and .ttft_spread_percent != null
                    and .ttft_spread_percent <= ($max_group_spread + 1e-9)
                  else true end))
              and ($lanes | length) == (2 * $width)
              and all($lanes[]; .samples == 2
                and .prompt_token_variants == 1
                and .wall_spread_percent <= ($max_case_spread + 1e-9)
                and (if $requires_latency then
                    .ttft_spread_percent != null
                    and .ttft_spread_percent <= ($max_case_spread + 1e-9)
                  else true end)
                and .completion_token_variants == 1
                and .scalar_parity == true)),
            observed_band_dominance:(
              ($hf2q.comparison_rates | min) >= ($reference.comparison_rates | max)
              and (if $requires_latency then
                  ($hf2q.wave_p95_wall | max) <= ($reference.wave_p95_wall | min)
                  and ($hf2q.wave_p95_ttft | max) <= ($reference.wave_p95_ttft | min)
                else true end)),
            maximum_group_spread_percent:$max_group_spread,
            maximum_case_spread_percent:$max_case_spread,
            engines:$engines,lanes:$lanes},
          hf2q_over_reference_comparison_rate:
            ($hf2q.median_comparison_rate / $reference.median_comparison_rate),
          reference_over_hf2q_p95_wall:
            ($reference.median_p95_wall / $hf2q.median_p95_wall),
          semantic_ttft:{required:$requires_latency,
            stable:(if $requires_latency then
                ($hf2q.ttft_spread_percent <= ($max_group_spread + 1e-9)
                 and $reference.ttft_spread_percent <= ($max_group_spread + 1e-9))
              else null end),
            observed_band_dominance:(if $requires_latency then
                ($hf2q.wave_p95_ttft | max) <= ($reference.wave_p95_ttft | min)
              else null end),
            reference_over_hf2q_p95:
              (if $requires_latency then
                ($reference.median_p95_ttft / $hf2q.median_p95_ttft)
               else null end),
            hf2q_p95_ms:$hf2q.median_p95_ttft,
            reference_p95_ms:$reference.median_p95_ttft},
          hf2q_median_comparison_rate:$hf2q.median_comparison_rate,
          reference_median_comparison_rate:$reference.median_comparison_rate,
          diagnostics:{
            hf2q_median_api_completion_tokens_per_second:
              $hf2q.median_api_completion_token_rate,
            reference_median_api_completion_tokens_per_second:
              $reference.median_api_completion_token_rate},
          hf2q_p95_wall_seconds:$hf2q.median_p95_wall,
          reference_p95_wall_seconds:$reference.median_p95_wall}
    ' "$rows_path"
}

matched_physical_validate_inner_summary() {
    local summary_path=$1
    jq -e \
      --argjson kv_budget "$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES" \
      --arg hf2q_speculation "$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY" \
      --arg reference_speculation "$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY" \
      --arg hf2q_kv "$QWEN38_MATCHED_HF2Q_KV_CACHE" \
      --arg reference_k "$QWEN38_MATCHED_REFERENCE_KV_CACHE_K" \
      --arg reference_v "$QWEN38_MATCHED_REFERENCE_KV_CACHE_V" \
      --argjson context_tokens "$QWEN38_MATCHED_CONTEXT_TOKENS" \
      --argjson decode_mvn "$QWEN38_PHYSICAL_DECODE_MVN" \
      --argjson decode_mv_ext "$QWEN38_PHYSICAL_DECODE_MV_EXT" \
      --argjson q5k_canonical_q4x4 \
        "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" \
      --argjson max_launch_skew \
        "$QWEN38_MATCHED_MAX_LAUNCH_SKEW_SECONDS" '
      . as $summary
      | .schema == 2 and .verdict == "pass"
      and .gate == "qwen38-matched-physical-abba"
      and (.harness.commit | test("^[0-9a-f]{40}$"))
      and .harness.source_binding == "clean exact harness worktree"
      and (.hf2q.commit | test("^[0-9a-f]{40}$"))
      and (.hf2q.binary_sha256 | test("^[0-9a-f]{64}$"))
      and .hf2q.effective_routing_policy == {
        dense_decode_mvn:$decode_mvn,dense_decode_mv_ext:$decode_mv_ext,
        dense_q5k_canonical_q4x4:$q5k_canonical_q4x4}
      and (.reference.commit | test("^[0-9a-f]{40}$"))
      and (.reference.binary_sha256 | test("^[0-9a-f]{64}$"))
      and (.reference.runtime_manifest_sha256 | test("^[0-9a-f]{64}$"))
      and .reference.expected_runtime_manifest_sha256
        == .reference.runtime_manifest_sha256
      and .reference.pin_policy == "observed-current-then-frozen"
      and .reference.frozen_for_run == true
      and (.reference.pin_file_sha256 | test("^[0-9a-f]{64}$"))
      and .reference.runtime_manifest_sha256
        == .evidence.reference_runtime_manifest_sha256
      and .reference.expected_runtime_manifest_sha256
        == .evidence.expected_reference_runtime_manifest_sha256
      and .reference.pin_file_sha256 == .evidence.reference_pin_file_sha256
      and (.physical_matrix_sha256 | test("^[0-9a-f]{64}$"))
      and .workload.widths == [1,2,4,8,16]
      and .workload.trial_order == ["hf2q","reference","reference","hf2q"]
      and .workload.speculation ==
        {hf2q:$hf2q_speculation,reference:$reference_speculation}
      and .workload.cache_settings == {
        hf2q:{format:$hf2q_kv,budget_bytes:$kv_budget,
          context_tokens_per_slot:$context_tokens},
        reference:{k_format:$reference_k,v_format:$reference_v,
          context_tokens_total:$context_tokens}}
      and .workload.scalar_replay_per_lane == true
      and .workload.reference_parallelism_matches_width == true
      and .workload.repeat_semantic_tokenization.unit
        == "canonical-semantic-output-token"
      and (.workload.repeat_semantic_tokenization.receipt_sha256
        | test("^[0-9a-f]{64}$"))
      and (.workload.repeat_semantic_tokenization.completion_tokens | type == "number"
        and . > 0 and . == floor)
      and .acceptance.minimum_hf2q_ratio >= 1
      and .acceptance.maximum_launch_skew_seconds == $max_launch_skew
      and .host_contention.policy == "process-group-cpu-v2"
      and .host_contention.maximum_foreign_cpu_percent == 100
      and .host_contention.owner_scope == "release-gate-process-group"
      and (.host_contention.owner_pgid | type == "number"
        and floor == . and . > 0)
      and .host_contention.continuous == true
      and (.results | length) == 5
      and ([.results[].width] | sort) == [1,2,4,8,16]
      and all(.results[]; . as $result
        | .schema == 2
        and .hf2q_effective_routing_policy
          == $summary.hf2q.effective_routing_policy
        and .acceptance.minimum_hf2q_ratio >= 1
        and .physical_proof.width == .width
        and .physical_proof.mode == "ordinary-target-speculation-off"
        and .physical_proof.seal_validated == true
        and (.physical_proof.clients | length) == .width
        and ([.physical_proof.clients[].lane] | sort)
          == [range(1; $result.width + 1)]
        and all(.physical_proof.clients[]; .scalar_parity == true)
        and .physical_proof.scheduler_max_width == .width
        and .physical_proof.target_body_max_width == .width
        and .physical_proof.target_head_max_width == .width
        and .physical_proof.command_buffer_submissions_delta > 0
        and .speculation.hf2q_policy == $hf2q_speculation
        and .speculation.reference_policy == $reference_speculation
        and (.speculation.waves | length) == 8
        and ([.speculation.waves[] | [.trial,.engine,.group]])
          == [[1,"hf2q","code"],[1,"hf2q","repeat"],
              [2,"reference","code"],[2,"reference","repeat"],
              [3,"reference","code"],[3,"reference","repeat"],
              [4,"hf2q","code"],[4,"hf2q","repeat"]]
        and all(.speculation.waves[];
          .schema == 1 and .proof_pass == true
          and .policy == (if .engine == "hf2q" then $hf2q_speculation
            else $reference_speculation end)
          and (.proposals | type == "number")
          and .proposals == (.proposals | floor) and .proposals > 0
          and (.drafted_tokens | type == "number")
          and .drafted_tokens == (.drafted_tokens | floor)
          and .drafted_tokens > 0
          and (.accepted_tokens | type == "number")
          and .accepted_tokens == (.accepted_tokens | floor)
          and .accepted_tokens >= 0
          and .accepted_tokens <= .drafted_tokens
          and (.cost_disabled_generations | type == "number")
          and .cost_disabled_generations == (.cost_disabled_generations | floor)
          and .cost_disabled_generations >= 0
          and (.measured_round_seconds | type == "number")
          and .measured_round_seconds >= 0
          and (.equivalent_ordinary_seconds | type == "number")
          and .equivalent_ordinary_seconds >= 0
          and (if .engine == "reference" then
              .proof_mode == "accepted-proposals"
              and .accepted_tokens > 0 and .disable_reason == null
            elif .engine == "hf2q" and .proof_mode == "accepted-proposals" then
              .accepted_tokens > 0 and .disable_reason == null
            elif .engine == "hf2q" and .proof_mode == "measured-cost-disabled" then
              .accepted_tokens == 0
              and .cost_disabled_generations > 0
              and .measured_round_seconds > 0
              and .equivalent_ordinary_seconds > 0
              and .disable_reason == "measured_cost_unprofitable"
            else false end))
        and .code.width == .width and .code.group == "code"
        and .code.measurement_consistent == true
        and .code.api_concurrency_pass == true
        and .code.token_accounting.pass == true
        and .code.token_accounting.cross_engine_prompt_equality_required == true
        and .code.token_accounting.raw_api_completion_equality_required_within_engine == true
        and .code.token_accounting.cross_engine_completion_equality_required == false
        and .code.token_accounting.cross_engine_semantic_completion_equality_required == false
        and .code.quality_pass == true
        and .code.stability.stable == true
        and .code.stability.observed_band_dominance == true
        and .code.hf2q_over_reference_comparison_rate
          >= .acceptance.minimum_hf2q_ratio
        and .repeat.width == .width and .repeat.group == "repeat"
        and .repeat.measurement_consistent == true
        and .repeat.api_concurrency_pass == true
        and .repeat.token_accounting.pass == true
        and .repeat.token_accounting.cross_engine_prompt_equality_required == true
        and .repeat.token_accounting.raw_api_completion_equality_required_within_engine == true
        and .repeat.token_accounting.cross_engine_completion_equality_required == false
        and .repeat.token_accounting.cross_engine_semantic_completion_equality_required == true
        and .repeat.token_accounting.semantic_tokenization_sha256
          == $summary.workload.repeat_semantic_tokenization.receipt_sha256
        and .repeat.quality_pass == true
        and .repeat.stability.stable == true
        and .repeat.stability.observed_band_dominance == true
        and .repeat.hf2q_over_reference_comparison_rate
          >= .acceptance.minimum_hf2q_ratio
        and .repeat.reference_over_hf2q_p95_wall
          >= .acceptance.minimum_hf2q_ratio
        and .repeat.semantic_ttft.required == true
        and .repeat.semantic_ttft.stable == true
        and .repeat.semantic_ttft.observed_band_dominance == true
        and .repeat.semantic_ttft.reference_over_hf2q_p95
          >= .acceptance.minimum_hf2q_ratio)
    ' "$summary_path" >/dev/null
}

matched_physical_validate_expected_reference_closure() {
    local summary_path=$1 expected_runtime_manifest_sha=$2
    [[ "$expected_runtime_manifest_sha" =~ ^[0-9a-f]{64}$ ]] || return 1
    jq -e --arg expected "$expected_runtime_manifest_sha" '
      .reference.runtime_manifest_sha256 == $expected
      and .reference.expected_runtime_manifest_sha256 == $expected
      and .evidence.reference_runtime_manifest_sha256 == $expected
      and .evidence.expected_reference_runtime_manifest_sha256 == $expected
    ' "$summary_path" >/dev/null
}

matched_physical_validate_matrix_reference_cohort() {
    local summary_path=$1 expected_runtime_manifest_sha=$2
    [[ "$expected_runtime_manifest_sha" =~ ^[0-9a-f]{64}$ ]] || return 1
    jq -e --arg expected "$expected_runtime_manifest_sha" \
      --argjson decode_mvn "$QWEN38_PHYSICAL_DECODE_MVN" \
      --argjson decode_mv_ext "$QWEN38_PHYSICAL_DECODE_MV_EXT" \
      --argjson q5k_canonical_q4x4 \
        "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" '
      .schema == 2
      and .hf2q_effective_routing_policy == {
        dense_decode_mvn:$decode_mvn,dense_decode_mv_ext:$decode_mv_ext,
        dense_q5k_canonical_q4x4:$q5k_canonical_q4x4}
      and (.harness.commit | test("^[0-9a-f]{40}$"))
      and .harness.source_binding == "clean exact harness worktree"
      and .reference_runtime_manifest_sha256 == $expected
      and (.results | length) == 5
      and ([.results[].harness.commit] | unique) == [.harness.commit]
      and ([.results[].reference.runtime_manifest_sha256] | unique)
        == [$expected]
      and ([.results[].reference.expected_runtime_manifest_sha256] | unique)
        == [$expected]
      and ([.results[].evidence.reference_runtime_manifest_sha256] | unique)
        == [$expected]
      and ([.results[].evidence.expected_reference_runtime_manifest_sha256]
        | unique) == [$expected]
      and ([.results[].hf2q.effective_routing_policy] | unique)
        == [.hf2q_effective_routing_policy]
      and ([.results[].reference.pin_file_sha256] | unique | length) == 1
      and ([.results[].evidence.reference_pin_file_sha256] | unique)
        == ([.results[].reference.pin_file_sha256] | unique)
    ' "$summary_path" >/dev/null
}

matched_physical_extract_proof_json() {
    local physical_matrix=$1
    local format=$2
    local width=$3
    jq -ce --arg format "$format" --argjson width "$width" '
      .results[] | select(.model.format == $format)
      | .results[] | select(.width == $width)
      | {width,mode:"ordinary-target-speculation-off",seal_validated:true,
          scheduler_max_width:.metrics.scheduler_max_width,
          target_body_max_width:.metrics.target_body_max_width,
          target_head_max_width:.metrics.target_head_max_width,
          command_buffers_created_delta:.metrics.command_buffers_created_delta,
          command_buffer_submissions_delta:.metrics.command_buffer_submissions_delta,
          clients:[.clients[] | {lane,scalar_parity}]}
    ' "$physical_matrix"
}

matched_physical_require_child_seal() {
    matched_require_result_seal "$1"
}

matched_physical_validate_reopened_child() {
    local child_dir=$1 expected_owner width trial engine trial_dir
    local width_count width_trial_count trial_count=0
    matched_physical_require_child_seal "$child_dir" || return 1
    matched_physical_validate_inner_summary "$child_dir/summary.json" || return 1
    expected_owner=$(jq -er '.host_contention.owner_pgid' \
      "$child_dir/summary.json") || return 1
    matched_validate_contention_preflight_log \
      "$child_dir/contention-preflight.tsv" 22 "$expected_owner" || return 1
    matched_require_evidence_manifest_entry "$child_dir" \
      "$child_dir/contention-preflight.tsv" || return 1
    [[ -d "$child_dir/widths" && ! -L "$child_dir/widths" ]] || return 1
    width_count=$(find "$child_dir/widths" -mindepth 1 -maxdepth 1 -type d \
      -name 'width-*' | wc -l | tr -d '[:space:]') || return 1
    [[ "$width_count" == 5 ]] || return 1
    for width in 1 2 4 8 16; do
        [[ -d "$child_dir/widths/width-$width" \
          && ! -L "$child_dir/widths/width-$width" \
          && -d "$child_dir/widths/width-$width/trials" \
          && ! -L "$child_dir/widths/width-$width/trials" ]] || return 1
        width_trial_count=$(find "$child_dir/widths/width-$width/trials" \
          -mindepth 1 -maxdepth 1 -type d -name 'trial-*' \
          | wc -l | tr -d '[:space:]') || return 1
        [[ "$width_trial_count" == 4 ]] || return 1
        trial=0
        for engine in hf2q reference reference hf2q; do
            trial=$((trial + 1))
            trial_dir="$child_dir/widths/width-$width/trials/trial-$trial-$engine"
            matched_validate_reopened_trial_calibration "$trial_dir" \
              "$expected_owner" 120 5 "$child_dir" || return 1
            trial_count=$((trial_count + 1))
        done
    done
    [[ "$trial_count" == 20 ]]
}

matched_physical_validate_reopened_matrix() {
    local matrix_dir=$1 child_path child_index=0 expected_path actual_summary
    matched_require_result_seal "$matrix_dir" || return 1
    [[ -d "$matrix_dir/artifacts" && ! -L "$matrix_dir/artifacts" ]] \
      || return 1
    qwen38_validate_matched_physical_matrix_receipt "$matrix_dir/summary.json" \
      || return 1
    for expected_path in artifacts/bf16 artifacts/q4_k_m artifacts/q5_k_m \
      artifacts/q6_k artifacts/q8_0; do
        child_path="$matrix_dir/$expected_path"
        matched_physical_validate_reopened_child "$child_path" || return 1
        actual_summary=$(jq -S -c . "$child_path/summary.json") || return 1
        [[ "$actual_summary" == "$(jq -S -c --argjson index "$child_index" \
          '.results[$index]' "$matrix_dir/summary.json")" ]] || return 1
        child_index=$((child_index + 1))
    done
    ((child_index == 5))
}
