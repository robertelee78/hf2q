#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERIFIER="$ROOT_DIR/scripts/verify_gemma4_wave_thermal_receipt.sh"
tmp=$(mktemp -d)
cleanup_contract() {
  rm -rf "$tmp"
}
trap cleanup_contract EXIT

mkdir -p "$tmp/agents"
for agent in 1 2 3 4; do
  printf '{"status":"pass","agent":%d}\n' "$agent" \
    >"$tmp/agents/agent-$agent.cold.json"
done
for offset in 0 5 10 15 20 25 30 35 40 45 50 55 60; do
  printf '%d\tnominal\tgemma-wave-1-settle\n' "$((1000 + offset))"
done >"$tmp/settle.log"
{
  printf '2000\tnominal\tgemma-wave-1-measurement-start\n'
  printf '2002\tnominal\tgemma-wave-1-measurement\n'
  printf '2004\tnominal\tgemma-wave-1-measurement-end\n'
} >"$tmp/measurement.log"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
cold_receipts_json=$(
  for receipt in "$tmp"/agents/agent-*.cold.json; do
    jq -n --arg name "$(basename "$receipt")" \
      --arg sha256 "$(sha256_file "$receipt")" \
      '{name:$name,sha256:$sha256}'
  done | jq -s 'sort_by(.name)'
)
jq -n --arg phase gemma-wave-1 \
  --arg settle_log_sha256 "$(sha256_file "$tmp/settle.log")" \
  --arg measurement_log_sha256 "$(sha256_file "$tmp/measurement.log")" \
  --argjson cold_receipts "$cold_receipts_json" \
  '{status:"pass",phase:$phase,required_state:"nominal",
    measurement_scope:"full-agent-wave",cold_receipts:$cold_receipts,
    settle_seconds:60,settle_duration_seconds:60,settle_samples:13,
    measurement_samples:3,measurement_duration_seconds:4,
    sample_interval_seconds:2,maximum_sample_gap_seconds:5,
    settle_sample_interval_seconds:5,maximum_settle_sample_gap_seconds:8,
    non_nominal_measurement_samples:0,settle_telemetry_gaps:0,
    telemetry_gaps:0,settle_log_sha256:$settle_log_sha256,
    measurement_log_sha256:$measurement_log_sha256}' >"$tmp/thermal.json"
jq -n --slurpfile thermal "$tmp/thermal.json" \
  '{status:"pass",wave_id:"wave1",thermal:$thermal[0]}' >"$tmp/receipt.json"

bash "$VERIFIER" 1 "$tmp/receipt.json" "$tmp/thermal.json" \
  "$tmp/measurement.log" "$tmp/settle.log" "$tmp/agents"

cp "$tmp/measurement.log" "$tmp/measurement.invalid.log"
sed -i.bak '2s/nominal/fair/' "$tmp/measurement.invalid.log"
if bash "$VERIFIER" 1 "$tmp/receipt.json" "$tmp/thermal.json" \
  "$tmp/measurement.invalid.log" "$tmp/settle.log" "$tmp/agents" \
  >/dev/null 2>&1; then
  echo "Gemma thermal verifier accepted tampered non-Nominal telemetry" >&2
  exit 1
fi

jq 'del(.thermal.measurement_scope)' "$tmp/receipt.json" \
  >"$tmp/receipt.invalid.json"
if bash "$VERIFIER" 1 "$tmp/receipt.invalid.json" "$tmp/thermal.json" \
  "$tmp/measurement.log" "$tmp/settle.log" "$tmp/agents" \
  >/dev/null 2>&1; then
  echo "Gemma thermal verifier accepted an envelope/summary mismatch" >&2
  exit 1
fi

mkdir -p "$tmp/agents-extra"
cp "$tmp"/agents/agent-*.cold.json "$tmp/agents-extra/"
: >"$tmp/agents-extra/agent-5.cold.json"
if bash "$VERIFIER" 1 "$tmp/receipt.json" "$tmp/thermal.json" \
  "$tmp/measurement.log" "$tmp/settle.log" "$tmp/agents-extra" \
  >/dev/null 2>&1; then
  echo "Gemma thermal verifier accepted an extra cold receipt" >&2
  exit 1
fi

echo "Gemma full-wave thermal contract passed"
