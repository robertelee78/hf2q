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
  '{status:"pass",phase:$phase,concurrent_agents:4,required_state:"nominal",
    measurement_scope:"full-agent-wave",cold_receipts:$cold_receipts,
    settle_seconds:60,settle_duration_seconds:60,settle_samples:13,
    measurement_samples:3,measurement_duration_seconds:4,
    sample_interval_seconds:2,maximum_sample_gap_seconds:5,
    settle_sample_interval_seconds:5,maximum_settle_sample_gap_seconds:8,
    non_nominal_measurement_samples:0,settle_telemetry_gaps:0,
    telemetry_gaps:0,settle_log_sha256:$settle_log_sha256,
    measurement_log_sha256:$measurement_log_sha256}' >"$tmp/thermal.json"
jq -n --slurpfile thermal "$tmp/thermal.json" \
  '{status:"pass",wave_id:"wave1",concurrent_agents:4,thermal:$thermal[0]}' \
  >"$tmp/receipt.json"

bash "$VERIFIER" 1 "$tmp/receipt.json" "$tmp/thermal.json" \
  "$tmp/measurement.log" "$tmp/settle.log" "$tmp/agents"

# The experimental eight-slot lane uses the same fail-closed receipt shape,
# with an exact phase/agent binding rather than a weaker ad-hoc temperature
# check.
mkdir -p "$tmp/agents8"
for agent in 1 2 3 4 5 6 7 8; do
  printf '{"status":"pass","agent":%d}\n' "$agent" \
    >"$tmp/agents8/agent-$agent.cold.json"
done
sed 's/gemma-wave-1/gemma-eight-slots/g' "$tmp/settle.log" \
  >"$tmp/settle8.log"
sed 's/gemma-wave-1/gemma-eight-slots/g' "$tmp/measurement.log" \
  >"$tmp/measurement8.log"
cold_receipts8_json=$(
  for receipt in "$tmp"/agents8/agent-*.cold.json; do
    jq -n --arg name "$(basename "$receipt")" \
      --arg sha256 "$(sha256_file "$receipt")" \
      '{name:$name,sha256:$sha256}'
  done | jq -s 'sort_by(.name)'
)
jq -n --arg phase gemma-eight-slots \
  --arg settle_log_sha256 "$(sha256_file "$tmp/settle8.log")" \
  --arg measurement_log_sha256 "$(sha256_file "$tmp/measurement8.log")" \
  --argjson cold_receipts "$cold_receipts8_json" \
  '{status:"pass",phase:$phase,concurrent_agents:8,required_state:"nominal",
    measurement_scope:"full-agent-wave",cold_receipts:$cold_receipts,
    settle_seconds:60,settle_duration_seconds:60,settle_samples:13,
    measurement_samples:3,measurement_duration_seconds:4,
    sample_interval_seconds:2,maximum_sample_gap_seconds:5,
    settle_sample_interval_seconds:5,maximum_settle_sample_gap_seconds:8,
    non_nominal_measurement_samples:0,settle_telemetry_gaps:0,
    telemetry_gaps:0,settle_log_sha256:$settle_log_sha256,
    measurement_log_sha256:$measurement_log_sha256}' >"$tmp/thermal8.json"
jq -n --slurpfile thermal "$tmp/thermal8.json" \
  '{status:"pass",wave_id:"eight-slots",concurrent_agents:8,thermal:$thermal[0]}' \
  >"$tmp/receipt8.json"

bash "$VERIFIER" eight-slots "$tmp/receipt8.json" "$tmp/thermal8.json" \
  "$tmp/measurement8.log" "$tmp/settle8.log" "$tmp/agents8"

mkdir -p "$tmp/agents8-missing"
cp "$tmp"/agents8/agent-{1,2,3,4,5,6,7}.cold.json "$tmp/agents8-missing/"
if bash "$VERIFIER" eight-slots "$tmp/receipt8.json" "$tmp/thermal8.json" \
  "$tmp/measurement8.log" "$tmp/settle8.log" "$tmp/agents8-missing" \
  >/dev/null 2>&1; then
  echo "Gemma thermal verifier accepted a missing N=8 cold receipt" >&2
  exit 1
fi

mkdir -p "$tmp/agents8-empty"
cp "$tmp"/agents8/agent-*.cold.json "$tmp/agents8-empty/"
: >"$tmp/agents8-empty/agent-8.cold.json"
empty_sha=$(sha256_file "$tmp/agents8-empty/agent-8.cold.json")
jq --arg sha "$empty_sha" \
  '(.cold_receipts[] | select(.name == "agent-8.cold.json") | .sha256) = $sha' \
  "$tmp/thermal8.json" >"$tmp/thermal8.empty-agent.json"
jq --slurpfile thermal "$tmp/thermal8.empty-agent.json" \
  '.thermal = $thermal[0]' "$tmp/receipt8.json" \
  >"$tmp/receipt8.empty-agent.json"
if bash "$VERIFIER" eight-slots "$tmp/receipt8.empty-agent.json" \
  "$tmp/thermal8.empty-agent.json" "$tmp/measurement8.log" \
  "$tmp/settle8.log" "$tmp/agents8-empty" >/dev/null 2>&1; then
  echo "Gemma thermal verifier accepted an empty N=8 cold receipt" >&2
  exit 1
fi

if bash "$VERIFIER" eight-slots "$tmp/receipt.json" "$tmp/thermal.json" \
  "$tmp/measurement.log" "$tmp/settle.log" "$tmp/agents" \
  >/dev/null 2>&1; then
  echo "Gemma thermal verifier accepted swapped four-slot evidence for N=8" >&2
  exit 1
fi

jq '.concurrent_agents = "8"' "$tmp/receipt8.json" \
  >"$tmp/receipt8.string-agents.json"
if bash "$VERIFIER" eight-slots "$tmp/receipt8.string-agents.json" \
  "$tmp/thermal8.json" "$tmp/measurement8.log" "$tmp/settle8.log" \
  "$tmp/agents8" >/dev/null 2>&1; then
  echo "Gemma thermal verifier accepted a string agent count" >&2
  exit 1
fi

cp "$tmp/measurement8.log" "$tmp/measurement8.invalid-phase.log"
sed -i.bak '2s/gemma-eight-slots/gemma-wave-1/' \
  "$tmp/measurement8.invalid-phase.log"
jq --arg sha "$(sha256_file "$tmp/measurement8.invalid-phase.log")" \
  '.measurement_log_sha256 = $sha' "$tmp/thermal8.json" \
  >"$tmp/thermal8.invalid-phase.json"
jq --slurpfile thermal "$tmp/thermal8.invalid-phase.json" \
  '.thermal = $thermal[0]' "$tmp/receipt8.json" \
  >"$tmp/receipt8.invalid-phase.json"
if bash "$VERIFIER" eight-slots "$tmp/receipt8.invalid-phase.json" \
  "$tmp/thermal8.invalid-phase.json" "$tmp/measurement8.invalid-phase.log" \
  "$tmp/settle8.log" "$tmp/agents8" >/dev/null 2>&1; then
  echo "Gemma thermal verifier accepted a wrong N=8 measurement phase" >&2
  exit 1
fi

mkdir -p "$tmp/agents8-extra"
cp "$tmp"/agents8/agent-*.cold.json "$tmp/agents8-extra/"
: >"$tmp/agents8-extra/agent-9.cold.json"
if bash "$VERIFIER" eight-slots "$tmp/receipt8.json" "$tmp/thermal8.json" \
  "$tmp/measurement8.log" "$tmp/settle8.log" "$tmp/agents8-extra" \
  >/dev/null 2>&1; then
  echo "Gemma thermal verifier accepted an extra N=8 cold receipt" >&2
  exit 1
fi

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
