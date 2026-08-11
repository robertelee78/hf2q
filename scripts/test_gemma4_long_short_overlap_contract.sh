#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HARNESS="$ROOT_DIR/scripts/test_gemma4_long_short_overlap.sh"
RELEASE_GATE="$ROOT_DIR/scripts/run_agentic_cache_release_gate.sh"

for command in awk bash grep; do
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
grep -qF 'CURL_MAX_TIME_SECONDS must be a positive integer' "$invalid_stderr"

if SERVER_PID=1 SERVER_LOG=/dev/null BINARY_PATH=/bin/true \
  BINARY_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  MODEL_PATH=/dev/null \
  MODEL_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
  OUT_DIR=/dev/null CURL_MAX_TIME_SECONDS=1 CANCELLATION_WAIT_SECONDS=0 \
  bash "$HARNESS" 2>"$invalid_stderr"; then
  echo "Gemma overlap accepted a non-positive cancellation wait" >&2
  exit 1
fi
grep -qF 'CANCELLATION_WAIT_SECONDS must be a positive integer' "$invalid_stderr"

# The literal shell variable is the contract: every long curl must read the
# validated runtime value instead of embedding a fixed timeout.
# shellcheck disable=SC2016
[[ "$(grep -cF -- '--max-time "$CURL_MAX_TIME_SECONDS"' "$HARNESS")" == 3 ]] || {
  echo "every long-running Gemma curl must use the validated timeout" >&2
  exit 1
}
if grep -qF -- '--max-time 900' "$HARNESS"; then
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

awk '
  /^run_gemma_wave\(\)/ { in_wave=1 }
  in_wave && /if \[\[ "\$agents" == 8 \]\]/ { in_eight=1 }
  in_eight && /wave_limits=\(MAX_COLD_TTFT_MS=40000 MAX_TOOL_RESULT_RESPONSE_MS=30000\)/ {
    limits=1
  }
  limits && /env "\$\{wave_limits\[@\]\}"/ { forwards=1 }
  forwards && /scripts\/test_full_context_agent_slots.sh/ { found=1; exit }
  END { exit(found ? 0 : 1) }
' "$RELEASE_GATE"

echo "Gemma release harness contract passed"
