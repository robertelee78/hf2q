#!/usr/bin/env bash
# Validate hf2q's concrete OpenAI-compatible SSE wire format and write the
# ordered JSON data payloads, excluding the terminal [DONE], as JSON Lines.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 INPUT_SSE OUTPUT_JSONL" >&2
  exit 2
fi

input=$1
output=$2

fail() {
  echo "OpenAI SSE validation: $*" >&2
  exit 1
}

for command in awk jq mktemp mv rm; do
  command -v "$command" >/dev/null || fail "missing required command: $command"
done
[[ -s "$input" && ! -L "$input" ]] || fail "input is missing, empty, or linked"
[[ ! -e "$output" || ( -f "$output" && ! -L "$output" ) ]] || \
  fail "output exists and is not a regular file"

tmp=$(mktemp "${TMPDIR:-/tmp}/hf2q-openai-sse.XXXXXX")
cleanup() { rm -f "$tmp"; }
trap cleanup EXIT

# hf2q emits one `data: ...` field per event, blank-line event delimiters,
# and exact empty-comment keepalives (`:`). Be intentionally stricter than a
# general-purpose EventSource parser: other fields, mixed events, unframed
# records, data after [DONE], and unterminated final events are release fails.
awk -v output="$tmp" '
  function reject(message) {
    print "OpenAI SSE validation: " message > "/dev/stderr"
    failed = 1
    exit 1
  }
  function finish_event() {
    if (!in_event) reject("empty or repeated event delimiter")
    if (comment_lines == 1 && data_lines == 0 && line_count == 1) {
      in_event = 0
      return
    }
    if (data_lines != 1 || comment_lines != 0 || line_count != 1) {
      reject("each non-comment event must contain exactly one data field")
    }
    if (payload == "[DONE]") {
      if (done_count != 0) reject("duplicate [DONE] event")
      done_count = 1
    } else {
      if (done_count != 0) reject("data event follows [DONE]")
      print payload >> output
      json_count++
    }
    in_event = 0
  }
  {
    if (sub(/\r$/, "", $0)) saw_crlf = 1
    else saw_lf = 1
    if (saw_crlf && saw_lf) reject("mixed line endings")
    if ($0 == "") {
      finish_event()
      next
    }
    if (done_count != 0) reject("content follows [DONE]")
    if (!in_event) {
      in_event = 1
      line_count = 0
      data_lines = 0
      comment_lines = 0
      payload = ""
    }
    line_count++
    if ($0 == ":") {
      comment_lines++
    } else if (substr($0, 1, 6) == "data: ") {
      data_lines++
      payload = substr($0, 7)
    } else {
      reject("unsupported or malformed SSE field")
    }
  }
  END {
    if (failed) exit 1
    if (in_event) reject("final SSE event is not blank-line terminated")
    if (done_count != 1) reject("stream must contain exactly one [DONE] event")
    if (json_count < 1) reject("stream contains no JSON data events")
  }
' "$input"

jq -e -s '
  length > 0
  and all(.[]; type == "object" and (has("error") | not))
  and all(.[];
    (.choices | type == "array" and length == 1)
    and .choices[0].index == 0)
  and all(.[0:-1][]; .choices[0].finish_reason == null)
  and .[-1].choices[0].finish_reason == "stop"
' "$tmp" >/dev/null || fail "JSON events must contain exactly one normal stop"

mv "$tmp" "$output"
trap - EXIT
