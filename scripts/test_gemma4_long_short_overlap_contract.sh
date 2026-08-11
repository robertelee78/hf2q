#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HARNESS="$ROOT_DIR/scripts/test_gemma4_long_short_overlap.sh"
RELEASE_GATE="$ROOT_DIR/scripts/run_agentic_cache_release_gate.sh"

for command in bash rg; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done

bash -n "$HARNESS" "$RELEASE_GATE"

invalid_stderr=$(mktemp)
trap 'rm -f "$invalid_stderr"' EXIT
if SERVER_PID=1 SERVER_LOG=/dev/null BINARY_PATH=/bin/true \
  BINARY_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  MODEL_PATH=/dev/null \
  MODEL_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  OUT_DIR=/dev/null CURL_MAX_TIME_SECONDS=0 \
  bash "$HARNESS" 2>"$invalid_stderr"; then
  echo "Gemma overlap accepted a non-positive curl timeout" >&2
  exit 1
fi
rg -q '^CURL_MAX_TIME_SECONDS must be a positive integer$' "$invalid_stderr"

if SERVER_PID=1 SERVER_LOG=/dev/null BINARY_PATH=/bin/true \
  BINARY_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  MODEL_PATH=/dev/null \
  MODEL_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  OUT_DIR=/dev/null CURL_MAX_TIME_SECONDS=1 CANCELLATION_WAIT_SECONDS=0 \
  bash "$HARNESS" 2>"$invalid_stderr"; then
  echo "Gemma overlap accepted a non-positive cancellation wait" >&2
  exit 1
fi
rg -q '^CANCELLATION_WAIT_SECONDS must be a positive integer$' "$invalid_stderr"

[[ "$(rg -c -- '--max-time "\$CURL_MAX_TIME_SECONDS"' "$HARNESS")" == 3 ]] || {
  echo "every long-running Gemma curl must use the validated timeout" >&2
  exit 1
}
if rg -q -- '--max-time 900' "$HARNESS"; then
  echo "Gemma overlap still contains a non-overridable 900-second timeout" >&2
  exit 1
fi

awk '
  /^run_gemma_release_gates\(\)/ { in_gemma=1 }
  in_gemma && /CURL_MAX_TIME_SECONDS=1800 CANCELLATION_WAIT_SECONDS=180/ { armed=1; next }
  armed && /scripts\/test_gemma4_long_short_overlap.sh/ { found=1; exit }
  armed && substr($0, length($0), 1) != "\\" { exit 1 }
  END { exit(found ? 0 : 1) }
' "$RELEASE_GATE"

echo "Gemma long/short overlap timeout contract passed"
