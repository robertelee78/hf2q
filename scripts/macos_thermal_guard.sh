#!/usr/bin/env bash

# Shared macOS thermal-state helpers for calibrated hardware gates. This file
# is sourced by release scripts and intentionally performs no work on import.

thermal_read_state() {
  local swift_bin=${HF2Q_THERMAL_SWIFT_BIN:-/usr/bin/swift}
  local state

  [[ -x "$swift_bin" ]] || {
    echo "thermal-state probe is not executable: $swift_bin" >&2
    return 1
  }
  state=$(
    "$swift_bin" -e '
      import Foundation
      switch ProcessInfo.processInfo.thermalState {
      case .nominal: print("nominal")
      case .fair: print("fair")
      case .serious: print("serious")
      case .critical: print("critical")
      @unknown default: print("unknown")
      }
    ' 2>/dev/null
  ) || {
    echo "failed to read macOS thermal state" >&2
    return 1
  }
  case "$state" in
    nominal|fair|serious|critical) ;;
    *)
      echo "malformed macOS thermal state: ${state:-<empty>}" >&2
      return 1
      ;;
  esac
  THERMAL_STATE=$state
}

thermal_sample() {
  local log_file=$1
  local phase=$2
  local sampled_at

  thermal_read_state || return 1
  sampled_at=$(date +%s) || return 1
  [[ "$sampled_at" =~ ^[0-9]+$ ]] || {
    echo "malformed thermal sample timestamp: $sampled_at" >&2
    return 1
  }
  printf '%s\t%s\t%s\n' "$sampled_at" "$THERMAL_STATE" "$phase" \
    >>"$log_file"
}

thermal_wait_for_nominal() {
  local log_file=$1
  local phase=$2
  local settle_seconds=$3
  local timeout_seconds=$4
  local sample_seconds=$5
  local deadline
  local nominal_since

  for value in "$settle_seconds" "$timeout_seconds" "$sample_seconds"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
      echo "thermal timing values must be non-negative integers" >&2
      return 2
    }
  done
  deadline=$((SECONDS + timeout_seconds))
  nominal_since=-1
  : >"$log_file"
  while :; do
    thermal_sample "$log_file" "$phase" || return 1
    if [[ "$THERMAL_STATE" == nominal ]]; then
      if ((nominal_since < 0)); then
        nominal_since=$SECONDS
      fi
      if ((SECONDS - nominal_since >= settle_seconds)); then
        return 0
      fi
    else
      nominal_since=-1
    fi
    if ((SECONDS >= deadline)); then
      echo "thermal state did not remain nominal for ${settle_seconds}s within ${timeout_seconds}s" >&2
      return 1
    fi
    sleep "$sample_seconds"
  done
}

thermal_monitor_nominal() {
  local log_file=$1
  local phase=$2
  local stop_file=$3
  local sample_seconds=$4

  [[ "$sample_seconds" =~ ^[0-9]+$ ]] || {
    echo "thermal sample interval must be a non-negative integer" >&2
    return 2
  }
  while [[ ! -e "$stop_file" ]]; do
    thermal_sample "$log_file" "$phase" || return 1
    if [[ "$THERMAL_STATE" != nominal ]]; then
      echo "calibrated phase $phase observed non-nominal thermal state: $THERMAL_STATE" >&2
      return 1
    fi
    sleep "$sample_seconds"
  done
}

thermal_validate_measurement_log() {
  local log_file=$1
  local maximum_gap_seconds=$2
  local stats

  [[ "$maximum_gap_seconds" =~ ^[0-9]+$ ]] || return 2
  stats=$(awk -F '\t' -v maximum="$maximum_gap_seconds" '
    BEGIN { invalid = 0; gaps = 0; non_nominal = 0 }
    {
      if (NF != 3 || $1 !~ /^[0-9]+$/) {
        invalid++
        next
      }
      samples++
      if ($2 != "nominal") non_nominal++
      if (samples == 1) first = $1
      if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
      previous = $1
      last = $1
    }
    END {
      duration = samples > 0 ? last - first : -1
      printf "%d\t%d\t%d\t%d\t%d\n", samples, duration,
        non_nominal, gaps, invalid
    }
  ' "$log_file") || return 1
  IFS=$'\t' read -r THERMAL_LOG_SAMPLES THERMAL_LOG_DURATION_SECONDS \
    THERMAL_LOG_NON_NOMINAL_SAMPLES THERMAL_LOG_GAPS THERMAL_LOG_INVALID_ROWS \
    <<<"$stats"
  ((THERMAL_LOG_SAMPLES >= 2)) \
    && ((THERMAL_LOG_DURATION_SECONDS > 0)) \
    && ((THERMAL_LOG_NON_NOMINAL_SAMPLES == 0)) \
    && ((THERMAL_LOG_GAPS == 0)) \
    && ((THERMAL_LOG_INVALID_ROWS == 0))
}

thermal_validate_settle_log() {
  local log_file=$1
  local required_seconds=$2
  local maximum_gap_seconds=$3
  local stats

  [[ "$required_seconds" =~ ^[0-9]+$ ]] || return 2
  [[ "$maximum_gap_seconds" =~ ^[0-9]+$ ]] || return 2
  stats=$(awk -F '\t' -v maximum="$maximum_gap_seconds" '
    BEGIN { invalid = 0; gaps = 0; nominal_since = -1 }
    {
      if (NF != 3 || $1 !~ /^[0-9]+$/ \
          || $2 !~ /^(nominal|fair|serious|critical)$/) {
        invalid++
        next
      }
      samples++
      if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
      previous = $1
      if ($2 != "nominal") {
        non_nominal++
        nominal_since = -1
      } else if (nominal_since < 0) {
        nominal_since = $1
      }
      last = $1
    }
    END {
      duration = nominal_since >= 0 ? last - nominal_since : -1
      printf "%d\t%d\t%d\t%d\t%d\n", samples, duration,
        non_nominal, gaps, invalid
    }
  ' "$log_file") || return 1
  IFS=$'\t' read -r THERMAL_LOG_SAMPLES THERMAL_LOG_DURATION_SECONDS \
    THERMAL_LOG_NON_NOMINAL_SAMPLES THERMAL_LOG_GAPS THERMAL_LOG_INVALID_ROWS \
    <<<"$stats"
  ((THERMAL_LOG_SAMPLES > 0)) \
    && ((THERMAL_LOG_DURATION_SECONDS >= required_seconds)) \
    && ((THERMAL_LOG_GAPS == 0)) \
    && ((THERMAL_LOG_INVALID_ROWS == 0))
}
