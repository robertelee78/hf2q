#!/usr/bin/env bash

# Pure validators shared by the ADR-049 Qwen mixed-workload runner, its
# independent receipt verifier, and model-free contract tests.  This file
# deliberately performs no server or model work.

qwen35_mixed_metric_value() {
  local file=$1 name=$2
  awk -v name="$name" '$1 == name {value=$2; found++}
    END {
      if (found != 1 || value !~ /^[0-9]+([.][0-9]+)?$/) exit 1
      print value
    }' "$file"
}

qwen35_mixed_validate_publication() {
  local publication=$1 expected_mtp=$2
  awk -v line="$publication" -v expected_mtp="$expected_mtp" 'BEGIN {
    if (line !~ /Qwen rectangular prefill published/) exit 1
    if (line !~ /lanes=4/) exit 1
    if (line !~ /checkpoint_at_end=true/) exit 1
    rows=line; sub(/^.*rows_per_lane=/,"",rows); sub(/ .*/,"",rows); rows += 0
    aggregate=line; sub(/^.*aggregate_rows=/,"",aggregate); sub(/ .*/,"",aggregate); aggregate += 0
    if (rows < 16 || rows > 128 || aggregate != 4 * rows) exit 1
    if (expected_mtp == "succeeded") {
      exit !(line ~ /mtp_prefill=true/ && line ~ /mtp_outcome=Succeeded/)
    }
    exit !(expected_mtp == "not-requested" && line ~ /mtp_prefill=false/ &&
      line ~ /mtp_outcome=NotRequested/)
  }'
}

qwen35_mixed_validate_power_log() {
  local mode=$1 label=$2
  awk -F '\t' -v mode="$mode" -v label="$label" '
    BEGIN {
      expected[1]=label "-before-launch"
      expected[2]=label "-loaded-warm"
      expected[3]=label "-measurement-start"
      expected[4]=label "-measurement-end"
      expected[5]=label "-after-shutdown"
    }
    NF != 5 || $1 !~ /^[0-9]+$/ || $2 != "ac" || $3 != mode ||
      $4 !~ /^[0-9]+$/ || $5 != expected[NR] || (NR > 1 && $1 < prior) {bad++}
    {count++; codes[$4]=1; prior=$1}
    END {exit !(count == 5 && bad == 0 && length(codes) == 1)}
  '
}

qwen35_mixed_semantic_trace_json() {
  local frames=$1 request_started=$2 wave_started=$3 wave_finished=$4
  jq -Rsc --argjson request_started "$request_started" \
    --argjson wave_started "$wave_started" --argjson wave_finished "$wave_finished" '
    def semantic:
      (.choices[0].delta // {}) as $delta
      | (($delta.content // "") | length) > 0
        or (($delta.reasoning_content // "") | length) > 0
        or (($delta.tool_calls // []) | length) > 0;
    def max_gap($values):
      if ($values | length) < 2 then 0
      else reduce range(1; $values | length) as $i
        (0; (($values[$i] - $values[$i - 1]) * 1000) as $gap
          | if $gap > . then $gap else . end)
      end;
    split("\n") | map(select(length > 0)) as $raw_frames
    | [$raw_frames[]
      | capture("^(?<at>[0-9]+[.][0-9]+)\\t(?<payload>.*)$")
      | .at |= tonumber] as $frames
    | if ($frames | length) != ($raw_frames | length)
        or ($frames | length) < 2
        or ([$frames[] | select(.payload == "[DONE]")] | length) != 1
        or $frames[-1].payload != "[DONE]"
        or any(range(1; $frames | length); $frames[.].at < $frames[. - 1].at)
      then error("invalid timestamped SSE frame stream") else . end
    | [$frames[] | select(.payload != "[DONE]")
        | . + {event:(.payload | fromjson)}] as $json_frames
    | [$json_frames[] | select(.event | semantic)] as $semantic
    | if ($semantic | length) == 0 then error("no semantic SSE frames") else . end
    | [$semantic[].at] as $semantic_at
    | {
        semantic_events:($semantic | length),
        request_started_at:$request_started,
        first_semantic_at:$semantic_at[0],
        last_semantic_at:$semantic_at[-1],
        terminal_at:$frames[-1].at,
        wave_started_at:$wave_started,
        wave_finished_at:$wave_finished,
        first_semantic_ms:(($semantic_at[0] - $request_started) * 1000),
        max_semantic_gap_ms:max_gap($semantic_at),
        decoder_wall_ms:(($frames[-1].at - $request_started) * 1000),
        semantic_before_wave:([ $semantic_at[] | select(. < $wave_started) ] | length),
        semantic_during_wave:([ $semantic_at[]
          | select(. >= $wave_started and . <= $wave_finished) ] | length),
        semantic_after_wave:([ $semantic_at[] | select(. > $wave_finished) ] | length)
      }
  ' "$frames"
}

qwen35_mixed_semantic_frame_count() {
  local frames=$1
  [[ -f "$frames" ]] || { printf '0\n'; return; }
  jq -Rsc '
    def semantic:
      (.choices[0].delta // {}) as $delta
      | (($delta.content // "") | length) > 0
        or (($delta.reasoning_content // "") | length) > 0
        or (($delta.tool_calls // []) | length) > 0;
    [split("\n")[] | select(length > 0)
      | capture("^[0-9]+[.][0-9]+\\t(?<payload>.*)$").payload
      | select(. != "[DONE]") | fromjson | select(semantic)] | length
  ' "$frames" 2>/dev/null || printf '0\n'
}

qwen35_mixed_validate_semantic_trace() {
  local trace=$1 minimum_events=$2 max_ttft_ms=$3 max_gap_ms=$4
  jq -e --argjson minimum_events "$minimum_events" \
    --argjson max_ttft_ms "$max_ttft_ms" --argjson max_gap_ms "$max_gap_ms" '
    .semantic_events >= $minimum_events
    and .request_started_at <= .first_semantic_at
    and .first_semantic_at < .wave_started_at
    and .wave_started_at < .wave_finished_at
    and .wave_finished_at < .terminal_at
    and .first_semantic_ms >= 0 and .first_semantic_ms <= $max_ttft_ms
    and .max_semantic_gap_ms >= 0 and .max_semantic_gap_ms <= $max_gap_ms
    and .semantic_before_wave > 0
    and .semantic_during_wave > 0
    and .semantic_after_wave > 0
  ' "$trace" >/dev/null
}

qwen35_mixed_canonical_unary_json() {
  local response=$1
  jq -S '{
    message:(.choices[0].message
      | {role,content,reasoning_content,tool_calls,refusal}),
    finish_reason:.choices[0].finish_reason,
    usage:(.usage
      | {prompt_tokens,completion_tokens,total_tokens,prompt_tokens_details})
  }' "$response"
}

qwen35_mixed_canonical_sse_json() {
  local events=$1
  jq -Ssc '
    [.[].choices[0].delta.content // empty] as $content
    | [.[].choices[0].delta.reasoning_content // empty] as $reasoning
    | [.[].choices[0].delta.tool_calls[]?] as $tools
    | [.[].choices[0].finish_reason // empty] as $finish
    | [ .[] | select(has("usage")) | .usage ][-1] as $usage
    | {
        message:{role:"assistant",content:($content | join("")),
          reasoning_content:($reasoning | join("")),tool_calls:$tools,
          refusal:null},
        finish_reason:$finish[-1],
        usage:($usage
          | {prompt_tokens,completion_tokens,total_tokens,prompt_tokens_details})
      }
  ' "$events"
}
