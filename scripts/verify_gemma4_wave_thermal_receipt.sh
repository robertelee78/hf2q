#!/usr/bin/env bash
set -euo pipefail

wave=${1:?wave is required}
receipt=${2:?wave receipt path is required}
summary=${3:?thermal summary path is required}
measurement_log=${4:?measurement log path is required}
settle_log=${5:?settle log path is required}
cold_receipt_dir=${6:?cold receipt directory is required}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$ROOT_DIR/scripts/macos_thermal_guard.sh"

[[ "$wave" == 1 || "$wave" == 2 ]] || {
  echo "Gemma thermal receipt wave must be 1 or 2" >&2
  exit 2
}
for path in "$receipt" "$summary" "$measurement_log" "$settle_log"; do
  [[ -s "$path" ]] || {
    echo "Gemma thermal receipt input is missing or empty: $path" >&2
    exit 1
  }
done
[[ -d "$cold_receipt_dir" ]] || {
  echo "Gemma cold receipt directory is missing: $cold_receipt_dir" >&2
  exit 1
}

phase="gemma-wave-$wave"
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
cold_receipt_count=$(find "$cold_receipt_dir" -maxdepth 1 -type f \
  -name 'agent-*.cold.json' -size +0c | wc -l | tr -d '[:space:]')
[[ "$cold_receipt_count" == 4 ]] || {
  echo "Gemma cold receipt directory must contain exactly four receipts" >&2
  exit 1
}

test "$(jq -er .wave_id "$receipt")" = "wave$wave"
test "$(jq -er .phase "$summary")" = "$phase"
jq -e --slurpfile thermal "$summary" '.thermal == $thermal[0]' \
  "$receipt" >/dev/null
jq -e --arg phase "$phase" '
  .status == "pass"
  and .phase == $phase
  and .required_state == "nominal"
  and .measurement_scope == "full-agent-wave"
  and (.cold_receipts | type) == "array"
  and (.cold_receipts | length) == 4
  and ([.cold_receipts[].name] | unique | length) == 4
  and all(.cold_receipts[];
    (.name | test("^agent-[1-4]\\.cold\\.json$"))
    and (.sha256 | type) == "string"
    and (.sha256 | test("^[0-9a-f]{64}$")))
  and (.settle_seconds | type) == "number" and .settle_seconds == 60
  and (.settle_duration_seconds | type) == "number"
  and .settle_duration_seconds >= .settle_seconds
  and (.settle_samples | type) == "number" and .settle_samples > 0
  and (.measurement_samples | type) == "number" and .measurement_samples >= 2
  and (.measurement_duration_seconds | type) == "number"
  and .measurement_duration_seconds > 0
  and (.sample_interval_seconds | type) == "number"
  and .sample_interval_seconds == 2
  and (.maximum_sample_gap_seconds | type) == "number"
  and .maximum_sample_gap_seconds == 5
  and (.settle_sample_interval_seconds | type) == "number"
  and .settle_sample_interval_seconds == 5
  and (.maximum_settle_sample_gap_seconds | type) == "number"
  and .maximum_settle_sample_gap_seconds == 8
  and (.non_nominal_measurement_samples | type) == "number"
  and .non_nominal_measurement_samples == 0
  and (.settle_telemetry_gaps | type) == "number"
  and .settle_telemetry_gaps == 0
  and (.telemetry_gaps | type) == "number" and .telemetry_gaps == 0
  and (.settle_log_sha256 | type) == "string"
  and (.settle_log_sha256 | test("^[0-9a-f]{64}$"))
  and (.measurement_log_sha256 | type) == "string"
  and (.measurement_log_sha256 | test("^[0-9a-f]{64}$"))
' "$summary" >/dev/null

test "$(sha256_file "$measurement_log")" = \
  "$(jq -er .measurement_log_sha256 "$summary")"
test "$(sha256_file "$settle_log")" = \
  "$(jq -er .settle_log_sha256 "$summary")"
while IFS=$'\t' read -r name expected_sha; do
  [[ "$name" =~ ^agent-[1-4]\.cold\.json$ ]]
  test "$(sha256_file "$cold_receipt_dir/$name")" = "$expected_sha"
done < <(jq -r '.cold_receipts[] | [.name, .sha256] | @tsv' "$summary")

thermal_validate_measurement_log "$measurement_log" 5
test "$THERMAL_LOG_SAMPLES" = "$(jq -er .measurement_samples "$summary")"
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .measurement_duration_seconds "$summary")"
test "$THERMAL_LOG_NON_NOMINAL_SAMPLES" = \
  "$(jq -er .non_nominal_measurement_samples "$summary")"
test "$THERMAL_LOG_GAPS" = "$(jq -er .telemetry_gaps "$summary")"
test "$(head -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
  "$phase-measurement-start"
test "$(tail -1 "$measurement_log" | awk -F '\t' '{print $3}')" = \
  "$phase-measurement-end"
awk -F '\t' -v phase="$phase-measurement" '
  NR > 1 && $3 != phase && $3 != phase "-end" { exit 1 }
' "$measurement_log"

thermal_validate_settle_log "$settle_log" 60 8
test "$THERMAL_LOG_SAMPLES" = "$(jq -er .settle_samples "$summary")"
test "$THERMAL_LOG_DURATION_SECONDS" = \
  "$(jq -er .settle_duration_seconds "$summary")"
test "$THERMAL_LOG_GAPS" = "$(jq -er .settle_telemetry_gaps "$summary")"
awk -F '\t' -v phase="$phase-settle" '$3 != phase { exit 1 }' "$settle_log"

echo "Gemma full-wave thermal receipt verified: wave $wave" >&2
