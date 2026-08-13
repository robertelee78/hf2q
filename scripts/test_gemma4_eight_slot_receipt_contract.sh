#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FILTER="$ROOT_DIR/scripts/gemma4_eight_slot_receipt.jq"
tmp=$(mktemp -d)
cleanup_contract() {
  rm -rf "$tmp"
}
trap cleanup_contract EXIT

jq -n '
  def agent($n): {
    status:"pass",agent:$n,prompt_tokens:7119,cold_cached_tokens:0,
    cached_tokens:7112,auto_cached_tokens:7112,continuation_cached_tokens:7112,
    cold_ttft_ms:(25000 + $n),cold_semantic_response_ms:(28000 + $n),
    tool_result_response_ms:(20000 + $n)};
  [range(1;9) | agent(.)] as $agents |
  {
    status:"pass",family:"gemma4",wave_id:"eight-slots",
    concurrent_agents:8,require_cold_first:1,agents:$agents,
    maximum_cold_ttft_ms:($agents | map(.cold_ttft_ms) | max),
    maximum_cold_semantic_response_ms:
      ($agents | map(.cold_semantic_response_ms) | max),
    maximum_tool_result_ms:($agents | map(.tool_result_response_ms) | max),
    thermal:{
      status:"pass",phase:"gemma-eight-slots",concurrent_agents:8,
      required_state:"nominal",measurement_scope:"full-agent-wave",
      settle_seconds:60,settle_duration_seconds:60,settle_samples:13,
      measurement_samples:3,measurement_duration_seconds:4,
      sample_interval_seconds:2,maximum_sample_gap_seconds:5,
      settle_sample_interval_seconds:5,maximum_settle_sample_gap_seconds:8,
      non_nominal_measurement_samples:0,settle_telemetry_gaps:0,
      telemetry_gaps:0,
      cold_receipts:[range(1;9) | {
        name:("agent-" + (.|tostring) + ".cold.json"),
        sha256:("a" * 64)}]
    }
  }
' >"$tmp/valid.json"

jq -e -f "$FILTER" "$tmp/valid.json" >/dev/null

assert_rejected() {
  local name=$1
  local filter=$2
  jq "$filter" "$tmp/valid.json" >"$tmp/$name.json"
  if jq -e -f "$FILTER" "$tmp/$name.json" >/dev/null 2>&1; then
    echo "Gemma N=8 validator accepted tamper: $name" >&2
    exit 1
  fi
}

assert_rejected missing-agent '.agents |= .[0:7]'
assert_rejected string-timing '.agents[0].cold_ttft_ms = "1"'
assert_rejected forged-aggregate \
  '.agents[0].cold_ttft_ms = 40001 | .maximum_cold_ttft_ms = 0'
assert_rejected missing-thermal 'del(.thermal)'
assert_rejected wrong-phase '.thermal.phase = "gemma-wave-1"'
assert_rejected missing-cold-binding '.thermal.cold_receipts |= .[0:7]'

echo "Gemma eight-slot receipt contract passed"
