#!/usr/bin/env bash
# Exact-artifact cumulative Qwen release gate. One continuously powered
# max-slots=4 server must survive the incident-shaped overlap, a warm agentic
# wave, two measured four-agent waves, and a final tiny control without
# command-buffer or label-object population growth.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:18081}"
SERVER_PID="${SERVER_PID:?set SERVER_PID to the canonical server process}"
SERVER_LOG="${SERVER_LOG:?set SERVER_LOG to the canonical server log}"
BINARY_PATH="${BINARY_PATH:?set BINARY_PATH to the packed hf2q binary}"
BINARY_SHA256="${BINARY_SHA256:?set BINARY_SHA256}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the exact Qwen GGUF}"
MODEL_SHA256="${MODEL_SHA256:?set MODEL_SHA256}"
FIXTURE_JSON="${FIXTURE_JSON:?set FIXTURE_JSON to the canonical long fixture}"
SHORT_FIXTURE_JSON="${SHORT_FIXTURE_JSON:?set SHORT_FIXTURE_JSON to the canonical short fixture}"
MAX_SLOTS="${MAX_SLOTS:-4}"
OUT_DIR="${OUT_DIR:-$(mktemp -d -t hf2q-qwen36-cumulative.XXXXXX)}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in curl jq rg sed awk date shasum wc ps lsof sort seq caffeinate pmset heap stat find; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ "$MAX_SLOTS" == 4 ]] || {
  echo "cumulative Qwen release gate requires MAX_SLOTS=4" >&2
  exit 2
}
[[ -x "$BINARY_PATH" && -f "$MODEL_PATH" && -f "$SERVER_LOG" ]] || exit 2
[[ -f "$FIXTURE_JSON" && -f "$SHORT_FIXTURE_JSON" ]] || exit 2
qwen36_require_empty_receipt_dir "$OUT_DIR"

sha256_file() { shasum -a 256 "$1" | awk '{ print $1 }'; }
[[ "$(sha256_file "$BINARY_PATH")" == "$BINARY_SHA256" ]] || exit 2
[[ "$(sha256_file "$MODEL_PATH")" == "$MODEL_SHA256" ]] || exit 2
qwen36_bind_server_process "$BASE_URL" "$SERVER_PID" "$BINARY_PATH" \
  "$MODEL_PATH" "$MAX_SLOTS"

cleanup() { qwen36_stop_power_guard; }
trap cleanup EXIT
qwen36_start_power_guard "$SERVER_PID" "$OUT_DIR/caffeinate.log"
qwen36_write_log_baseline "$SERVER_LOG" "$OUT_DIR/server-log-baseline.json"

capture_heap() {
  local phase="$1"
  qwen36_assert_power_guard
  qwen36_capture_heap_summary "$SERVER_PID" \
    "$OUT_DIR/heap-$phase.txt" "$OUT_DIR/heap-$phase.json"
}

run_agent_wave() {
  local phase="$1"
  local max_tool_result_ms="${2:-10000}"
  local wave_dir="$OUT_DIR/$phase"
  mkdir -p "$wave_dir"
  qwen36_assert_power_guard
  BASE_URL="$BASE_URL" FAMILY=qwen36 AGENTS=4 \
    WAVE_ID="$phase" \
    REQUIRE_COLD_FIRST=1 \
    MAX_TOOL_RESULT_RESPONSE_MS="$max_tool_result_ms" \
    OUT_DIR="$wave_dir/agents" \
    "$script_dir/test_full_context_agent_slots.sh" >"$wave_dir/summary.json.tmp" \
    2>"$wave_dir/gate.err"
  jq -e '.status == "pass" and .family == "qwen36" and .concurrent_agents == 4 and .require_cold_first == 1 and all(.agents[]; .cold_cached_tokens == 0)' \
    "$wave_dir/summary.json.tmp" >/dev/null
  mv "$wave_dir/summary.json.tmp" "$wave_dir/summary.json"
  shasum -a 256 "$wave_dir/summary.json" >"$wave_dir/summary.json.sha256"
  shasum -c "$wave_dir/summary.json.sha256" >/dev/null
}

capture_heap baseline

overlap_dir="$OUT_DIR/overlap"
BASE_URL="$BASE_URL" SERVER_PID="$SERVER_PID" SERVER_LOG="$SERVER_LOG" \
  BINARY_PATH="$BINARY_PATH" BINARY_SHA256="$BINARY_SHA256" \
  FIXTURE_MODEL="$MODEL_PATH" MODEL_SHA256="$MODEL_SHA256" \
  FIXTURE_JSON="$FIXTURE_JSON" SHORT_FIXTURE_JSON="$SHORT_FIXTURE_JSON" \
  MAX_SLOTS=4 REQUIRE_PROVENANCE=1 OUT_DIR="$overlap_dir" \
  "$script_dir/test_qwen36_prefill_watchdog.sh" >"$OUT_DIR/overlap.stdout"
jq -e '.status == "pass" and .max_slots == 4 and .chunks == 44 and .full_chunks == 42 and .stable_tail_tokens > 0 and .generation_cue_tokens > 0 and .ready_http == 200' \
  "$overlap_dir/long-watchdog-summary.json" >/dev/null
shasum -c "$overlap_dir/long-watchdog-summary.json.sha256" >/dev/null
capture_heap post-overlap

# The first wave after the 87,972-token incident request deliberately rebuilds
# the four per-slot agentic working sets. It is a semantic/cache warmup, not a
# measured steady-state wave. Preserve a finite bound based on the observed
# 12-second cold transition, while keeping both measured waves on the existing
# 10-second Qwen tool-result contract.
run_agent_wave np4-warmup 15000
capture_heap post-warmup
run_agent_wave np4-wave1 10000
capture_heap post-wave1
run_agent_wave np4-wave2 10000
capture_heap post-wave2

qwen36_validate_heap_series \
  "$OUT_DIR/heap-baseline.json" "$OUT_DIR/heap-post-overlap.json" \
  "$OUT_DIR/heap-post-warmup.json" "$OUT_DIR/heap-post-wave1.json" \
  "$OUT_DIR/heap-post-wave2.json"

curl --fail-with-body --silent --show-error --no-buffer --connect-timeout 5 --max-time 60 \
  -H 'Content-Type: application/json' --data-binary "@$SHORT_FIXTURE_JSON" \
  -o "$OUT_DIR/final-tiny.sse" \
  -w 'http_code=%{http_code}\ntotal_seconds=%{time_total}\n' \
  "$BASE_URL/v1/chat/completions" >"$OUT_DIR/final-tiny.metrics" \
  2>"$OUT_DIR/final-tiny.err"
grep -qx 'http_code=200' "$OUT_DIR/final-tiny.metrics"
qwen36_extract_and_validate_sse final-tiny "$OUT_DIR/final-tiny.sse" \
  "$OUT_DIR/final-tiny-events.jsonl"
qwen36_validate_short_events "$OUT_DIR/final-tiny-events.jsonl"
ready_code=$(curl --silent --show-error --max-time 3 -o "$OUT_DIR/final-readyz.txt" \
  -w '%{http_code}' "$BASE_URL/readyz")
[[ "$ready_code" == 200 ]] || {
  echo "Qwen worker was not ready after cumulative release sequence" >&2
  exit 1
}
qwen36_bind_server_process "$BASE_URL" "$SERVER_PID" "$BINARY_PATH" \
  "$MODEL_PATH" "$MAX_SLOTS"
qwen36_assert_power_guard

server_delta="$OUT_DIR/server-log-delta.log"
qwen36_extract_append_only_log_delta "$SERVER_LOG" \
  "$OUT_DIR/server-log-baseline.json" "$server_delta"
qwen36_reject_fatal_log "$server_delta"

manifest_tmp="$OUT_DIR/artifact-manifest.sha256.tmp"
: >"$manifest_tmp"
for artifact in \
  "$overlap_dir/long-watchdog-summary.json" \
  "$overlap_dir/long-watchdog-summary.json.sha256" \
  "$overlap_dir/short-response.sse" \
  "$overlap_dir/long-response.sse" \
  "$overlap_dir/post-response.sse" \
  "$overlap_dir/short-events.jsonl" \
  "$overlap_dir/long-events.jsonl" \
  "$overlap_dir/post-events.jsonl" \
  "$overlap_dir/long-chunks.log" \
  "$overlap_dir/server-log-delta.log" \
  "$OUT_DIR/np4-warmup/summary.json" \
  "$OUT_DIR/np4-warmup/summary.json.sha256" \
  "$OUT_DIR/np4-wave1/summary.json" \
  "$OUT_DIR/np4-wave1/summary.json.sha256" \
  "$OUT_DIR/np4-wave2/summary.json" \
  "$OUT_DIR/np4-wave2/summary.json.sha256" \
  "$OUT_DIR/server-log-baseline.json" \
  "$OUT_DIR/heap-baseline.txt" \
  "$OUT_DIR/heap-baseline.json" \
  "$OUT_DIR/heap-post-overlap.txt" \
  "$OUT_DIR/heap-post-overlap.json" \
  "$OUT_DIR/heap-post-warmup.txt" \
  "$OUT_DIR/heap-post-warmup.json" \
  "$OUT_DIR/heap-post-wave1.txt" \
  "$OUT_DIR/heap-post-wave1.json" \
  "$OUT_DIR/heap-post-wave2.txt" \
  "$OUT_DIR/heap-post-wave2.json" \
  "$OUT_DIR/final-tiny.sse" \
  "$server_delta"; do
  [[ -f "$artifact" ]] || {
    echo "cumulative gate is missing a required receipt artifact: $artifact" >&2
    exit 1
  }
  shasum -a 256 "$artifact" >>"$manifest_tmp"
done
mv "$manifest_tmp" "$OUT_DIR/artifact-manifest.sha256"
shasum -c "$OUT_DIR/artifact-manifest.sha256" >/dev/null

summary="$OUT_DIR/cumulative-release-summary.json"
jq -n \
  --arg status pass \
  --arg binary_path "$BINARY_PATH" \
  --arg binary_sha256 "$BINARY_SHA256" \
  --arg model_path "$MODEL_PATH" \
  --arg model_sha256 "$MODEL_SHA256" \
  --arg manifest_sha256 "$(sha256_file "$OUT_DIR/artifact-manifest.sha256")" \
  --argjson server_pid "$SERVER_PID" \
  --argjson max_slots "$MAX_SLOTS" \
  --argjson warmup_max_tool_result_ms 15000 \
  --argjson measured_max_tool_result_ms 10000 \
  --argjson ready_http "$ready_code" \
  --argjson power_event_baseline "$QWEN36_POWER_EVENT_BASELINE" \
  --argjson power_event_final "$QWEN36_POWER_EVENT_FINAL" \
  --slurpfile baseline "$OUT_DIR/heap-baseline.json" \
  --slurpfile overlap "$OUT_DIR/heap-post-overlap.json" \
  --slurpfile warmup "$OUT_DIR/heap-post-warmup.json" \
  --slurpfile wave1 "$OUT_DIR/heap-post-wave1.json" \
  --slurpfile wave2 "$OUT_DIR/heap-post-wave2.json" \
  --slurpfile overlap_receipt "$overlap_dir/long-watchdog-summary.json" \
  --slurpfile warmup_receipt "$OUT_DIR/np4-warmup/summary.json" \
  --slurpfile wave1_receipt "$OUT_DIR/np4-wave1/summary.json" \
  --slurpfile wave2_receipt "$OUT_DIR/np4-wave2/summary.json" \
  '{status:$status,binary_path:$binary_path,binary_sha256:$binary_sha256,model_path:$model_path,model_sha256:$model_sha256,server_pid:$server_pid,max_slots:$max_slots,warmup_max_tool_result_ms:$warmup_max_tool_result_ms,measured_max_tool_result_ms:$measured_max_tool_result_ms,ready_http:$ready_http,power_event_baseline:$power_event_baseline,power_event_final:$power_event_final,power_event_delta:($power_event_final-$power_event_baseline),artifact_manifest_sha256:$manifest_sha256,heap_bounds_valid:true,overlap:$overlap_receipt[0],agent_waves:{warmup:$warmup_receipt[0],wave1:$wave1_receipt[0],wave2:$wave2_receipt[0]},heap_deltas:{cfstring_baseline_to_overlap:($overlap[0].cfstring_count-$baseline[0].cfstring_count),cfstring_overlap_to_warmup:($warmup[0].cfstring_count-$overlap[0].cfstring_count),cfstring_baseline_to_warmup:($warmup[0].cfstring_count-$baseline[0].cfstring_count),cfstring_warmup_to_wave1:($wave1[0].cfstring_count-$warmup[0].cfstring_count),cfstring_wave1_to_wave2:($wave2[0].cfstring_count-$wave1[0].cfstring_count),cfstring_warmup_to_wave2:($wave2[0].cfstring_count-$warmup[0].cfstring_count),pool_baseline_to_overlap:($overlap[0].autoreleasepool_content_count-$baseline[0].autoreleasepool_content_count),pool_overlap_to_warmup:($warmup[0].autoreleasepool_content_count-$overlap[0].autoreleasepool_content_count),pool_baseline_to_warmup:($warmup[0].autoreleasepool_content_count-$baseline[0].autoreleasepool_content_count),pool_warmup_to_wave1:($wave1[0].autoreleasepool_content_count-$warmup[0].autoreleasepool_content_count),pool_wave1_to_wave2:($wave2[0].autoreleasepool_content_count-$wave1[0].autoreleasepool_content_count),pool_warmup_to_wave2:($wave2[0].autoreleasepool_content_count-$warmup[0].autoreleasepool_content_count)},heap:{baseline:$baseline[0],post_overlap:$overlap[0],post_warmup:$warmup[0],post_wave1:$wave1[0],post_wave2:$wave2[0]}}' \
  >"$summary.tmp"
jq -e '.status == "pass" and .server_pid > 0 and .max_slots == 4 and .warmup_max_tool_result_ms == 15000 and .measured_max_tool_result_ms == 10000 and .ready_http == 200 and .power_event_delta == 0 and .overlap.status == "pass" and .overlap.prompt_tokens == 87972 and .overlap.chunks == 44 and .overlap.full_chunks == 42 and all(.agent_waves[]; .status == "pass" and .family == "qwen36" and .concurrent_agents == 4 and .require_cold_first == 1 and all(.agents[]; .cold_cached_tokens == 0)) and .heap_bounds_valid == true and all(.heap[]; .command_buffer_objects == 0 and .command_buffer_impls == 0) and .heap_deltas.cfstring_baseline_to_overlap <= 512 and .heap_deltas.cfstring_overlap_to_warmup <= 512 and .heap_deltas.cfstring_baseline_to_warmup <= 1024 and .heap_deltas.cfstring_warmup_to_wave1 <= 256 and .heap_deltas.cfstring_wave1_to_wave2 <= 256 and .heap_deltas.cfstring_warmup_to_wave2 <= 512 and .heap_deltas.pool_baseline_to_overlap <= 8 and .heap_deltas.pool_overlap_to_warmup <= 8 and .heap_deltas.pool_baseline_to_warmup <= 16 and .heap_deltas.pool_warmup_to_wave1 <= 8 and .heap_deltas.pool_wave1_to_wave2 <= 8 and .heap_deltas.pool_warmup_to_wave2 <= 16' \
  "$summary.tmp" >/dev/null
mv "$summary.tmp" "$summary"
shasum -a 256 "$summary" >"$summary.sha256"
shasum -c "$summary.sha256" >/dev/null
cat "$summary"
