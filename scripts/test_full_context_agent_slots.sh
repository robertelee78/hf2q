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
# cold DeepSeek requests share one 100 GiB verifier. The checked-in 21,204-byte
# context fixture preserves the 6,684-token insertion-ordered workload used for
# the exact matched peer comparison. Two thermally valid llama.cpp np4 waves
# measured 68.438 s
# and 69.944 s on this M5 Max; the 60 s hf2q ceiling keeps a 9.2 s margin below
# the peer median. Cached turns remain tightly bounded.
# Explicit operator values always win.
if [[ "$FAMILY" == deepseek4 && "$AGENTS" -gt 1 ]]; then
  MAX_COLD_TTFT_MS=${MAX_COLD_TTFT_MS:-60000}
  MAX_COLD_RESPONSE_MS=${MAX_COLD_RESPONSE_MS:-60000}
  MAX_CACHED_RESPONSE_MS=${MAX_CACHED_RESPONSE_MS:-15000}
  MAX_CACHED_SEMANTIC_MS=${MAX_CACHED_SEMANTIC_MS:-15000}
  MAX_TOOL_RESULT_RESPONSE_MS=${MAX_TOOL_RESULT_RESPONSE_MS:-35000}
  CURL_MAX_TIME_SECONDS=${CURL_MAX_TIME_SECONDS:-90}
  export MAX_COLD_TTFT_MS MAX_COLD_RESPONSE_MS MAX_CACHED_RESPONSE_MS
  export MAX_CACHED_SEMANTIC_MS MAX_TOOL_RESULT_RESPONSE_MS CURL_MAX_TIME_SECONDS
fi
cold_wait_seconds=${CURL_MAX_TIME_SECONDS:-60}
for command in jq curl; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ -x /usr/bin/perl ]] || {
  echo "required monotonic timer is unavailable: /usr/bin/perl" >&2
  exit 2
}
[[ -x /usr/bin/pgrep ]] || {
  echo "required child-process enumerator is unavailable: /usr/bin/pgrep" >&2
  exit 2
}
[[ -x "$GATE" ]] || {
  echo "agentic gate is not executable: $GATE" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
pids=()
agent_receipts=()
monotonic_us() {
  /usr/bin/perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
    -e 'printf "%.0f\n", 1000000 * clock_gettime(CLOCK_MONOTONIC)'
}
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
}
terminate_running_agents() {
  local child
  local pid
  local -a descendants=()

  # Freeze live gate roots first so they cannot spawn more curl/jq children
  # while the failure path enumerates the process tree.
  for pid in "${pids[@]:-}"; do
    kill -STOP "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]:-}"; do
    while IFS= read -r child; do
      [[ -n "$child" ]] || continue
      kill -STOP "$child" 2>/dev/null || true
      descendants+=("$child")
    done < <(/usr/bin/pgrep -P "$pid" 2>/dev/null || true)
  done
  for child in "${descendants[@]:-}"; do
    kill -KILL "$child" 2>/dev/null || true
  done
  for pid in "${pids[@]:-}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

cohort_started_us=$(monotonic_us)
for ((agent = 1; agent <= AGENTS; agent++)); do
  if [[ "$WAVE_ID" == default ]]; then
    request_run_id="full-context-${FAMILY}-agent-${agent}"
  else
    request_run_id="full-context-${FAMILY}-${WAVE_ID}-agent-${agent}"
  fi
  RUN_ID="$request_run_id" \
  REQUIRE_COLD_FIRST="$REQUIRE_COLD_FIRST" \
  COLD_RESULT_PATH="$OUT_DIR/agent-${agent}.cold.json" \
  SENTINEL="HF2Q_${FAMILY_TAG}_AGENT_${agent}_OK" \
    "$GATE" >"$OUT_DIR/agent-${agent}.json" 2>"$OUT_DIR/agent-${agent}.err" &
  pids+=("$!")
  agent_receipts+=("$OUT_DIR/agent-${agent}.json")
done

failed=0
cold_deadline=$((SECONDS + cold_wait_seconds + 30))
while :; do
  cold_receipts=$(find "$OUT_DIR" -maxdepth 1 -name 'agent-*.cold.json' | wc -l | tr -d '[:space:]')
  if (( cold_receipts == AGENTS )); then
    cohort_finished_us=$(monotonic_us)
    cohort_cold_wall_ms=$(( \
      (cohort_finished_us - cohort_started_us + 999) / 1000 \
    ))
    break
  fi
  for ((agent = 1; agent <= AGENTS; agent++)); do
    if [[ ! -s "$OUT_DIR/agent-${agent}.cold.json" ]] \
      && ! kill -0 "${pids[$((agent - 1))]}" 2>/dev/null; then
      echo "agent $agent exited before publishing its cold receipt" >&2
      failed=1
      cohort_cold_wall_ms=-1
      break 2
    fi
  done
  if (( SECONDS >= cold_deadline )); then
    echo "full-context cold cohort did not finish within $((cold_wait_seconds + 30)) seconds" >&2
    failed=1
    cohort_cold_wall_ms=-1
    break
  fi
  sleep 0.1
done
if ((failed != 0)); then
  terminate_running_agents
fi
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
  --argjson cohort_cold_wall_ms "$cohort_cold_wall_ms" \
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
      fixture_id: (map(.fixture_id) | unique
        | if length == 1 then .[0] else error("agentic fixture ids differ") end),
      agentic_context_fixture_sha256: (map(.agentic_context_fixture_sha256) | unique
        | if length == 1 then .[0] else error("agentic context fixture digests differ") end),
      agentic_context_fixture_bytes: (map(.agentic_context_fixture_bytes) | unique
        | if length == 1 then .[0] else error("agentic context fixture sizes differ") end),
      repository_context_chars: (map(.repository_context_chars) | unique
        | if length == 1 then .[0] else error("agentic context character counts differ") end),
      agentic_system_prompt_sha256: (map(.agentic_system_prompt_sha256) | unique
        | if length == 1 then .[0] else error("agentic system prompts differ") end),
      tool_result_success_prefix_sha256: (map(.tool_result_success_prefix_sha256) | unique
        | if length == 1 then .[0] else error("tool-result success envelopes differ") end),
      expected_prompt_tokens: (map(.expected_prompt_tokens) | unique
        | if length == 1 then .[0] else error("expected prompt token counts differ") end),
      prompt_tokens: (map(.prompt_tokens) | unique
        | if length == 1 then .[0] else error("agent prompt token counts differ") end),
      maximum_cold_ttft_ms: (map(.cold_ttft_ms) | max),
      maximum_cold_semantic_response_ms: (map(.cold_semantic_response_ms) | max),
      cohort_cold_wall_ms: $cohort_cold_wall_ms,
      minimum_cached_tokens: (map(.cached_tokens) | min),
      maximum_cached_ttft_ms: (map(.cached_ttft_ms) | max),
      maximum_tool_result_ms: (map(.tool_result_response_ms) | max),
      agents: .
    }
  end
' "${agent_receipts[@]}"

echo "full-context agent-slot receipts: $OUT_DIR" >&2
