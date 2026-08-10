#!/usr/bin/env bash
set -euo pipefail

# Run the complete OpenCode acceptance contract from several independent
# conversations at the same time. The server is started separately so this
# harness can be used against the canonical family launchers and matched peers.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FAMILY=${FAMILY:-qwen36}
AGENTS=${AGENTS:-4}
OUT_DIR=${OUT_DIR:-$(mktemp -d -t hf2q-full-context-slots.XXXXXX)}
WAVE_ID=${WAVE_ID:-default}
REQUIRE_COLD_FIRST=${REQUIRE_COLD_FIRST:-1}

case "$FAMILY" in
  qwen36)
    GATE="$ROOT_DIR/scripts/test_qwen36_agentic.sh"
    FAMILY_TAG=QWEN36
    ;;
  gemma4)
    GATE="$ROOT_DIR/scripts/test_gemma4_agentic.sh"
    FAMILY_TAG=GEMMA4
    ;;
  deepseek4)
    GATE="$ROOT_DIR/scripts/test_deepseek4_agentic.sh"
    FAMILY_TAG=DEEPSEEK4
    ;;
  *)
    echo "FAMILY must be qwen36, gemma4, or deepseek4 (got: $FAMILY)" >&2
    exit 2
    ;;
esac

if ! [[ "$AGENTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "AGENTS must be a positive integer (got: $AGENTS)" >&2
  exit 2
fi
[[ "$WAVE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "WAVE_ID contains unsupported characters: $WAVE_ID" >&2
  exit 2
}
[[ "$REQUIRE_COLD_FIRST" == 0 || "$REQUIRE_COLD_FIRST" == 1 ]] || {
  echo "REQUIRE_COLD_FIRST must be 0 or 1" >&2
  exit 2
}

# The single-agent gate's 40-second cold limit is intentionally strict. Four
# cold DeepSeek requests share one 100 GiB verifier, so apply the independently
# measured llama.cpp np4 wall-clock bound (50.71 s on this M5 Max) plus a small
# scheduling margin. Cached turns remain tightly bounded. Explicit operator
# values always win.
if [[ "$FAMILY" == deepseek4 && "$AGENTS" -gt 1 ]]; then
  MAX_COLD_TTFT_MS=${MAX_COLD_TTFT_MS:-55000}
  MAX_COLD_RESPONSE_MS=${MAX_COLD_RESPONSE_MS:-55000}
  MAX_CACHED_RESPONSE_MS=${MAX_CACHED_RESPONSE_MS:-15000}
  MAX_CACHED_SEMANTIC_MS=${MAX_CACHED_SEMANTIC_MS:-15000}
  MAX_TOOL_RESULT_RESPONSE_MS=${MAX_TOOL_RESULT_RESPONSE_MS:-35000}
  CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-90}
  export MAX_COLD_TTFT_MS MAX_COLD_RESPONSE_MS MAX_CACHED_RESPONSE_MS
  export MAX_CACHED_SEMANTIC_MS MAX_TOOL_RESULT_RESPONSE_MS CURL_MAX_TIME_SECONDS
fi
for command in jq curl; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ -x "$GATE" ]] || {
  echo "agentic gate is not executable: $GATE" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM EXIT

for ((agent = 1; agent <= AGENTS; agent++)); do
  RUN_ID="full-context-${FAMILY}-${WAVE_ID}-agent-${agent}" \
  REQUIRE_COLD_FIRST="$REQUIRE_COLD_FIRST" \
  SENTINEL="HF2Q_${FAMILY_TAG}_AGENT_${agent}_OK" \
    "$GATE" >"$OUT_DIR/agent-${agent}.json" 2>"$OUT_DIR/agent-${agent}.err" &
  pids+=("$!")
done

failed=0
for ((agent = 1; agent <= AGENTS; agent++)); do
  if ! wait "${pids[$((agent - 1))]}"; then
    failed=1
    echo "agent $agent failed; stderr follows:" >&2
    sed -n '1,160p' "$OUT_DIR/agent-${agent}.err" >&2
  fi
done
pids=()
trap - INT TERM EXIT

if ((failed != 0)); then
  echo "full-context agent-slot gate failed; receipts: $OUT_DIR" >&2
  exit 1
fi

jq -s --arg family "$FAMILY" --arg wave_id "$WAVE_ID" \
  --argjson require_cold_first "$REQUIRE_COLD_FIRST" --argjson agents "$AGENTS" '
  if length != $agents or any(.[]; .status != "pass") then
    error("one or more agent receipts did not pass")
  else
    {
      status: "pass",
      family: $family,
      wave_id: $wave_id,
      require_cold_first: $require_cold_first,
      concurrent_agents: $agents,
      minimum_cached_tokens: (map(.cached_tokens) | min),
      maximum_cached_ttft_ms: (map(.cached_ttft_ms) | max),
      maximum_tool_result_ms: (map(.tool_result_response_ms) | max),
      agents: .
    }
  end
' "$OUT_DIR"/agent-*.json

echo "full-context agent-slot receipts: $OUT_DIR" >&2
