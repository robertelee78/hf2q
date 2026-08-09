#!/usr/bin/env bash
# Shared fail-closed validators for the Qwen 3.6 long-prefill hardware gates.
# This file is sourced by the live harness and its model-free negative tests.

QWEN36_FATAL_LOG_PATTERN='GPU Timeout|SubmissionsIgnored|Command buffer error|Generation error|engine_unhealthy|panicked at|worker-fatal'

qwen36_require_empty_receipt_dir() {
  local receipt_dir="$1"
  if [[ ! -e "$receipt_dir" ]]; then
    mkdir -p "$receipt_dir"
    return
  fi
  [[ -d "$receipt_dir" ]] || {
    echo "Qwen receipt path is not a directory: $receipt_dir" >&2
    return 1
  }
  [[ -z "$(find "$receipt_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "Qwen release receipt directory must be fresh and empty: $receipt_dir" >&2
    return 1
  }
}

qwen36_reject_fatal_log() {
  local log_path="$1"
  [[ -f "$log_path" && -r "$log_path" ]] || {
    echo "Qwen gate cannot read the required server log: $log_path" >&2
    return 1
  }
  if grep -Eiq "$QWEN36_FATAL_LOG_PATTERN" "$log_path"; then
    echo "Qwen gate observed a fatal worker/GPU log signature" >&2
    return 1
  fi
}

qwen36_file_identity() {
  local path="$1"
  stat -f '%d:%i' "$path" 2>/dev/null || stat -c '%d:%i' "$path" 2>/dev/null
}

qwen36_log_prefix_sha256() {
  local log_path="$1"
  local line_count="$2"
  awk -v limit="$line_count" 'NR <= limit' "$log_path" \
    | shasum -a 256 | awk '{ print $1 }'
}

qwen36_write_log_baseline() {
  local log_path="$1"
  local baseline_path="$2"
  local identity line_count prefix_sha256

  [[ -f "$log_path" && -r "$log_path" ]] || return 1
  identity=$(qwen36_file_identity "$log_path") || return 1
  line_count=$(wc -l <"$log_path" | tr -d ' ')
  [[ "$line_count" =~ ^[0-9]+$ ]] || return 1
  prefix_sha256=$(qwen36_log_prefix_sha256 "$log_path" "$line_count") || return 1
  jq -n --arg identity "$identity" --argjson line_count "$line_count" \
    --arg prefix_sha256 "$prefix_sha256" \
    '{identity:$identity,line_count:$line_count,prefix_sha256:$prefix_sha256}' \
    >"$baseline_path.tmp"
  mv "$baseline_path.tmp" "$baseline_path"
}

qwen36_extract_append_only_log_delta() {
  local log_path="$1"
  local baseline_path="$2"
  local delta_path="$3"
  local expected_identity baseline_lines expected_prefix current_identity final_lines current_prefix

  [[ -f "$baseline_path" && -r "$baseline_path" ]] || return 1
  expected_identity=$(jq -er '.identity' "$baseline_path") || return 1
  baseline_lines=$(jq -er '.line_count' "$baseline_path") || return 1
  expected_prefix=$(jq -er '.prefix_sha256' "$baseline_path") || return 1
  current_identity=$(qwen36_file_identity "$log_path") || return 1
  [[ "$current_identity" == "$expected_identity" ]] || {
    echo "server log rotated during Qwen gate" >&2
    return 1
  }
  final_lines=$(wc -l <"$log_path" | tr -d ' ')
  [[ "$final_lines" =~ ^[0-9]+$ ]] && (( final_lines >= baseline_lines )) || {
    echo "server log was truncated during Qwen gate" >&2
    return 1
  }
  current_prefix=$(qwen36_log_prefix_sha256 "$log_path" "$baseline_lines") || return 1
  [[ "$current_prefix" == "$expected_prefix" ]] || {
    echo "server log baseline changed during Qwen gate" >&2
    return 1
  }
  sed -n "$((baseline_lines + 1)),\$p" "$log_path" >"$delta_path.tmp"
  mv "$delta_path.tmp" "$delta_path"
}

qwen36_sse_data_bytes() {
  local sse_path="$1"
  awk '/^data: / { bytes += length($0) + 1 } END { print bytes + 0 }' "$sse_path"
}

qwen36_reject_successful_terminal_sse() {
  local sse_path="$1"

  [[ -f "$sse_path" && -r "$sse_path" ]] || {
    echo "cancelled SSE receipt is missing or unreadable: $sse_path" >&2
    return 1
  }

  if grep -Fxq 'data: [DONE]' "$sse_path"; then
    echo "cancelled SSE unexpectedly reached [DONE]" >&2
    return 1
  fi
  sed -n 's/^data: //p' "$sse_path" \
    | jq -se '
        ((map(.id) | unique | length) <= 1) and
        all(.[];
          type == "object" and
          (.id | type == "string" and length > 0) and
          .object == "chat.completion.chunk" and
          (.choices | type == "array" and length == 1) and
          (.choices[0].delta | type == "object") and
          .choices[0].finish_reason == null
        )
      ' >/dev/null || {
      echo "cancelled SSE contained malformed data, an error object, or a successful terminal finish" >&2
      return 1
    }
}

qwen36_parse_heap_summary() {
  local heap_path="$1"
  local json_path="$2"
  local parsed cf_count cf_bytes cb_objects cb_impls pool_count pool_bytes

  parsed=$(awk '
    function number(value, normalized) {
      if (value !~ /^[0-9]+$/ && value !~ /^[0-9][0-9]?[0-9]?(,[0-9][0-9][0-9])+$/) {
        invalid = 1
        return 0
      }
      normalized = value
      gsub(/,/, "", normalized)
      return normalized + 0
    }
    $4 == "CFString" && $5 == "ObjC" {
      if (++cf_rows != 1) exit 2
      cf_count = number($1)
      cf_bytes = number($2)
    }
    $4 ~ /(FamilyCommandBuffer|IOGPUMetalCommandBuffer)/ &&
      $4 !~ /[Ss]torage[Pp]ool/ {
      if ($4 ~ /_impl/) cb_impls += number($1)
      else cb_objects += number($1)
    }
    $4 == "@autoreleasepool" && $5 == "content" && $6 == "C" {
      if (++pool_rows != 1) exit 3
      pool_count = number($1)
      pool_bytes = number($2)
    }
    END {
      if (invalid || cf_rows != 1 || pool_rows != 1) exit 4
      printf "%d\t%d\t%d\t%d\t%d\t%d\n", \
        cf_count, cf_bytes, cb_objects, cb_impls, pool_count, pool_bytes
    }
  ' "$heap_path") || {
    echo "heap summary is missing or duplicates a required population row: $heap_path" >&2
    return 1
  }
  IFS=$'\t' read -r cf_count cf_bytes cb_objects cb_impls pool_count pool_bytes \
    <<<"$parsed"
  jq -n \
    --argjson cfstring_count "$cf_count" \
    --argjson cfstring_bytes "$cf_bytes" \
    --argjson command_buffer_objects "$cb_objects" \
    --argjson command_buffer_impls "$cb_impls" \
    --argjson autoreleasepool_content_count "$pool_count" \
    --argjson autoreleasepool_content_bytes "$pool_bytes" \
    '{cfstring_count:$cfstring_count,cfstring_bytes:$cfstring_bytes,command_buffer_objects:$command_buffer_objects,command_buffer_impls:$command_buffer_impls,autoreleasepool_content_count:$autoreleasepool_content_count,autoreleasepool_content_bytes:$autoreleasepool_content_bytes}' \
    >"$json_path.tmp"
  jq -e '.cfstring_count >= 0 and .command_buffer_objects >= 0 and .command_buffer_impls >= 0 and .autoreleasepool_content_count >= 0' \
    "$json_path.tmp" >/dev/null
  mv "$json_path.tmp" "$json_path"
}

qwen36_capture_heap_summary() {
  local server_pid="$1"
  local raw_path="$2"
  local json_path="$3"

  command -v heap >/dev/null || {
    echo "Qwen release gate requires /usr/bin/heap" >&2
    return 1
  }
  [[ "$server_pid" =~ ^[0-9]+$ ]] && kill -0 "$server_pid" 2>/dev/null || {
    echo "cannot capture heap from non-live server PID: $server_pid" >&2
    return 1
  }
  heap -q "$server_pid" >"$raw_path" 2>"$raw_path.err" &
  local heap_pid=$!
  local heap_deadline=$(( $(date +%s) + 20 ))
  while kill -0 "$heap_pid" 2>/dev/null; do
    if (( $(date +%s) >= heap_deadline )); then
      kill "$heap_pid" 2>/dev/null || true
      wait "$heap_pid" 2>/dev/null || true
      echo "heap timed out for server PID $server_pid" >&2
      return 1
    fi
    sleep 0.1
  done
  wait "$heap_pid" || {
    echo "heap failed for server PID $server_pid" >&2
    return 1
  }
  [[ ! -s "$raw_path.err" ]] || {
    echo "heap wrote diagnostics for server PID $server_pid" >&2
    return 1
  }
  qwen36_parse_heap_summary "$raw_path" "$json_path"
}

qwen36_validate_heap_series() {
  local baseline="$1"
  local post_overlap="$2"
  local post_warmup="$3"
  local post_wave1="$4"
  local post_wave2="$5"

  jq -e -s '
    length == 5 and
    all(.[]; .command_buffer_objects == 0 and .command_buffer_impls == 0) and
    all(.[]; .cfstring_count >= 0 and .autoreleasepool_content_count >= 0) and
    (.[1].cfstring_count - .[0].cfstring_count <= 512) and
    (.[2].cfstring_count - .[1].cfstring_count <= 512) and
    (.[2].cfstring_count - .[0].cfstring_count <= 1024) and
    (.[3].cfstring_count - .[2].cfstring_count <= 256) and
    (.[4].cfstring_count - .[3].cfstring_count <= 256) and
    (.[4].cfstring_count - .[2].cfstring_count <= 512) and
    (.[1].autoreleasepool_content_count - .[0].autoreleasepool_content_count <= 8) and
    (.[2].autoreleasepool_content_count - .[1].autoreleasepool_content_count <= 8) and
    (.[2].autoreleasepool_content_count - .[0].autoreleasepool_content_count <= 16) and
    (.[3].autoreleasepool_content_count - .[2].autoreleasepool_content_count <= 8) and
    (.[4].autoreleasepool_content_count - .[3].autoreleasepool_content_count <= 8) and
    (.[4].autoreleasepool_content_count - .[2].autoreleasepool_content_count <= 16)
  ' "$baseline" "$post_overlap" "$post_warmup" "$post_wave1" "$post_wave2" \
    >/dev/null || {
      echo "Qwen cumulative heap populations exceeded the fail-closed lifetime bounds" >&2
      return 1
    }
}

qwen36_power_event_count() {
  pmset -g log \
    | awk '/Entering Sleep state|ThermalEvent/ { count++ } END { print count + 0 }'
}

qwen36_validate_power_event_counts() {
  local baseline="$1"
  local final="$2"
  [[ "$baseline" =~ ^[0-9]+$ && "$final" =~ ^[0-9]+$ ]] || {
    echo "Qwen gate power-event counters must be non-negative integers" >&2
    return 1
  }
  [[ "$final" == "$baseline" ]] || {
    echo "Qwen gate was interrupted by sleep or a thermal event (power events $baseline -> $final)" >&2
    return 1
  }
}

qwen36_start_power_guard() {
  local target_pid="$1"
  local guard_log="$2"
  local attempt

  command -v caffeinate >/dev/null || {
    echo "Qwen hardware gate requires caffeinate" >&2
    return 1
  }
  command -v pmset >/dev/null || {
    echo "Qwen hardware gate requires pmset" >&2
    return 1
  }
  [[ "$target_pid" =~ ^[0-9]+$ ]] && kill -0 "$target_pid" 2>/dev/null || {
    echo "Qwen hardware gate power-guard target PID is not live: $target_pid" >&2
    return 1
  }
  pmset -g batt | grep -Fq "Now drawing from 'AC Power'" || {
    echo "Qwen hardware gate requires AC power" >&2
    return 1
  }

  QWEN36_POWER_EVENT_BASELINE=$(qwen36_power_event_count)
  caffeinate -dimsu -w "$target_pid" >"$guard_log" 2>&1 &
  QWEN36_POWER_GUARD_PID=$!
  for attempt in $(seq 1 50); do
    if ! kill -0 "$QWEN36_POWER_GUARD_PID" 2>/dev/null; then
      wait "$QWEN36_POWER_GUARD_PID" 2>/dev/null || true
      echo "Qwen hardware gate caffeinate guard exited during startup" >&2
      return 1
    fi
    if pmset -g assertions \
      | grep -Eq "pid ${QWEN36_POWER_GUARD_PID}\\(caffeinate\\)"; then
      return 0
    fi
    sleep 0.02
  done

  kill "$QWEN36_POWER_GUARD_PID" 2>/dev/null || true
  wait "$QWEN36_POWER_GUARD_PID" 2>/dev/null || true
  QWEN36_POWER_GUARD_PID=
  echo "Qwen hardware gate could not verify the caffeinate assertion" >&2
  return 1
}

qwen36_assert_power_guard() {
  [[ -n "${QWEN36_POWER_GUARD_PID:-}" ]] \
    && kill -0 "$QWEN36_POWER_GUARD_PID" 2>/dev/null || {
      echo "Qwen hardware gate lost its caffeinate assertion" >&2
      return 1
    }
  pmset -g batt | grep -Fq "Now drawing from 'AC Power'" || {
    echo "Qwen hardware gate lost AC power" >&2
    return 1
  }
  QWEN36_POWER_EVENT_FINAL=$(qwen36_power_event_count)
  qwen36_validate_power_event_counts \
    "$QWEN36_POWER_EVENT_BASELINE" "$QWEN36_POWER_EVENT_FINAL"
}

qwen36_stop_power_guard() {
  if [[ -n "${QWEN36_POWER_GUARD_PID:-}" ]]; then
    kill "$QWEN36_POWER_GUARD_PID" 2>/dev/null || true
    wait "$QWEN36_POWER_GUARD_PID" 2>/dev/null || true
    QWEN36_POWER_GUARD_PID=
  fi
}

qwen36_bind_server_process() {
  local base_url="$1"
  local server_pid="$2"
  local binary_path="$3"
  local model_path="$4"
  local max_slots="$5"
  local port server_command listener_pids

  [[ "$base_url" =~ ^http://(127\.0\.0\.1|localhost):([0-9]+)$ ]] || {
    echo "Qwen gate requires a loopback http BASE_URL with an explicit port: $base_url" >&2
    return 1
  }
  port="${BASH_REMATCH[2]}"
  [[ "$server_pid" =~ ^[0-9]+$ && -x "$binary_path" && -f "$model_path" ]] || {
    echo "Qwen gate requires a live SERVER_PID plus exact binary/model paths" >&2
    return 1
  }
  command -v lsof >/dev/null || {
    echo "Qwen gate requires lsof to bind BASE_URL to SERVER_PID" >&2
    return 1
  }
  listener_pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | sort -u)
  [[ "$listener_pids" == "$server_pid" ]] || {
    echo "BASE_URL port $port is not owned exclusively by SERVER_PID $server_pid (got: ${listener_pids:-none})" >&2
    return 1
  }
  server_command=$(ps -ww -p "$server_pid" -o command=)
  [[ "$server_command" == "$binary_path" || "$server_command" == "$binary_path "* ]] || {
    echo "SERVER_PID does not execute BINARY_PATH: $server_command" >&2
    return 1
  }
  [[ " $server_command " == *" --model $model_path "* ]] || {
    echo "SERVER_PID does not serve MODEL_PATH: $server_command" >&2
    return 1
  }
  [[ " $server_command " == *" --port $port "* ]] || {
    echo "SERVER_PID argv does not match BASE_URL port $port" >&2
    return 1
  }
  [[ " $server_command " == *" --max-slots $max_slots "* ]] || {
    echo "SERVER_PID argv does not prove --max-slots $max_slots" >&2
    return 1
  }
}

qwen36_validate_chunk_lines() {
  local chunk_lines="$1"
  [[ "$(wc -l <"$chunk_lines" | tr -d ' ')" == 43 ]] || {
    echo "expected exactly 43 request-scoped long chunks" >&2
    return 1
  }
  qwen36_validate_chunk_prefix_lines "$chunk_lines" 43
}

qwen36_validate_chunk_prefix_lines() {
  local chunk_lines="$1"
  local expected_count="$2"
  [[ "$expected_count" =~ ^[0-9]+$ && "$expected_count" -ge 1 && "$expected_count" -le 43 ]] || {
    echo "invalid expected Qwen chunk prefix count: $expected_count" >&2
    return 1
  }
  [[ "$(wc -l <"$chunk_lines" | tr -d ' ')" == "$expected_count" ]] || {
    echo "expected exactly $expected_count request-scoped long chunks" >&2
    return 1
  }
  awk -v expected_count="$expected_count" '
    {
      delete field
      delete seen
      for (i = 1; i <= NF; i++) {
        delete pair
        parts = split($i, pair, "=")
        if (pair[1] ~ /^(slot|chunk_start|chunk_end|chunk_tokens|prompt_tokens)$/) {
          if (parts != 2 || pair[2] !~ /^[0-9]+$/ || ++seen[pair[1]] != 1) {
            print "invalid bounded-chunk field at line " NR ": " $i > "/dev/stderr"
            exit 1
          }
          field[pair[1]] = pair[2] + 0
        }
      }
      if (!("slot" in field) ||
          !("chunk_start" in field) ||
          !("chunk_end" in field) ||
          !("chunk_tokens" in field) ||
          !("prompt_tokens" in field)) {
        print "missing required bounded-chunk field at line " NR ": " $0 > "/dev/stderr"
        exit 1
      }
      if (NR == 1) expected_slot = field["slot"]
      expected_start = (NR - 1) * 2048
      expected_tokens = (NR <= 42 ? 2048 : 1956)
      if (field["slot"] != expected_slot ||
          field["chunk_start"] != expected_start ||
          field["chunk_end"] != expected_start + expected_tokens ||
          field["chunk_tokens"] != expected_tokens ||
          field["prompt_tokens"] != 87972) {
        print "invalid bounded chunk at line " NR ": " $0 > "/dev/stderr"
        exit 1
      }
    }
    END { if (NR != expected_count) exit 1 }
  ' "$chunk_lines"
}

qwen36_extract_and_validate_sse() {
  local name="$1"
  local sse="$2"
  local json_stream="$3"
  local done_count
  awk 'NF && $0 !~ /^data: / && $0 !~ /^:/ { exit 1 }' "$sse" || {
    echo "$name SSE contained a non-data, non-comment event line" >&2
    return 1
  }
  [[ "$(sed -n 's/^data: //p' "$sse" | awk 'NF { value=$0 } END { print value }')" == "[DONE]" ]] || {
    echo "$name SSE did not terminate with [DONE]" >&2
    return 1
  }
  done_count=$(sed -n 's/^data: //p' "$sse" \
    | awk '$0 == "[DONE]" { count++ } END { print count + 0 }')
  [[ "$done_count" == 1 ]] || {
    echo "$name expected one SSE [DONE], observed $done_count" >&2
    return 1
  }
  sed -n 's/^data: //p' "$sse" \
    | awk '$0 != "[DONE]" && $0 != ""' >"$json_stream"
  [[ -s "$json_stream" ]] || {
    echo "$name SSE contained no JSON events" >&2
    return 1
  }
  jq -e . "$json_stream" >/dev/null || return 1
  ! jq -e 'select(.choices[0].finish_reason == "error")' "$json_stream" >/dev/null
}

qwen36_validate_short_events() {
  local json_stream="$1"
  local content finish_count tool_count
  content=$(jq -cs '[.[] | .choices[0].delta.content // empty] | join("")' "$json_stream")
  finish_count=$(jq -r '.choices[0].finish_reason // empty' "$json_stream" \
    | awk '$0 == "stop" { count++ } END { print count + 0 }')
  tool_count=$(jq -s '[.[] | .choices[0].delta.tool_calls[]?] | length' "$json_stream")
  [[ "$content" == '"OK"' ]] || {
    echo "short lane content was not exact OK: $content" >&2
    return 1
  }
  [[ "$finish_count" == 1 && "$tool_count" == 0 ]] || {
    echo "short lane must have one stop finish and no tool calls" >&2
    return 1
  }
  jq -se '
    ([.[] | .id] | unique | length) == 1 and
    all(.[];
      ((.id // "") | length) > 0 and
      .object == "chat.completion.chunk" and
      (((.choices // []) | length) == 1 or
       (((.choices // []) | length) == 0 and has("usage"))) and
      (if ((.choices // []) | length) == 1 then
         ((.choices[0].finish_reason // "stop") == "stop")
       else true end))
  ' "$json_stream" >/dev/null
}

qwen36_validate_long_events() {
  local json_stream="$1"
  local tool_name tool_args finish_count
  tool_name=$(jq -cs '[.[] | .choices[0].delta.tool_calls[]?.function.name // empty] | join("")' "$json_stream")
  tool_args=$(jq -cs '[.[] | .choices[0].delta.tool_calls[]?.function.arguments // empty] | join("")' "$json_stream")
  finish_count=$(jq -r '.choices[0].finish_reason // empty' "$json_stream" \
    | awk '$0 == "tool_calls" { count++ } END { print count + 0 }')
  [[ "$tool_name" == '"fixture_tool_346"' ]] || {
    echo "unexpected tool name: $tool_name" >&2
    return 1
  }
  [[ "$tool_args" == '"{\"path\":\"src/serve/api/engine.rs\"}"' ]] || {
    echo "unexpected tool arguments: $tool_args" >&2
    return 1
  }
  [[ "$finish_count" == 1 ]] || {
    echo "expected one tool_calls finish, observed $finish_count" >&2
    return 1
  }
  jq -se '
    ([.[] | .id] | unique | length) == 1 and
    ([.[] | .choices[0].delta.tool_calls[]?] | length) > 0 and
    ([.[] | .choices[0].delta.tool_calls[]?.index] | unique) == [0] and
    ([.[] | .choices[0].delta.tool_calls[]?.id? |
      select(type == "string" and length > 0)] | unique | length) == 1 and
    ([.[] | .choices[0].delta.tool_calls[]?.type? |
      select(type == "string" and length > 0)] | unique) == ["function"] and
    all(.[]; ((.choices[0].delta.content // "") | length) == 0) and
    all(.[];
      ((.id // "") | length) > 0 and
      .object == "chat.completion.chunk" and
      (((.choices // []) | length) == 1 or
       (((.choices // []) | length) == 0 and has("usage"))) and
      (if ((.choices // []) | length) == 1 then
         ((.choices[0].finish_reason // "tool_calls") == "tool_calls")
       else true end))
  ' "$json_stream" >/dev/null
}
