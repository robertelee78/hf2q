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

thermal_read_process_state() {
  local producer_pid=$1
  local state

  if ! state=$(/bin/ps -p "$producer_pid" -o state= 2>/dev/null \
    | tr -d '[:space:]'); then
    if kill -0 "$producer_pid" 2>/dev/null; then
      echo "failed to read live thermal producer state: $producer_pid" >&2
      return 1
    fi
    THERMAL_PROCESS_STATE=""
    return 0
  fi
  if [[ -z "$state" ]] && kill -0 "$producer_pid" 2>/dev/null; then
    echo "thermal producer state was empty for live pid: $producer_pid" >&2
    return 1
  fi
  THERMAL_PROCESS_STATE=$state
}

thermal_monitor_nominal_while_pid() {
  local log_file=$1
  local phase=$2
  local producer_pid=$3
  local sample_seconds=$4

  [[ "$producer_pid" =~ ^[1-9][0-9]*$ ]] || {
    echo "thermal producer pid must be a positive integer" >&2
    return 2
  }
  [[ "$sample_seconds" =~ ^[0-9]+$ ]] || {
    echo "thermal sample interval must be a non-negative integer" >&2
    return 2
  }
  while :; do
    thermal_read_process_state "$producer_pid" || return 1
    if [[ -z "$THERMAL_PROCESS_STATE" || "$THERMAL_PROCESS_STATE" == Z* ]]; then
      return 0
    fi
    thermal_sample "$log_file" "$phase" || return 1
    if [[ "$THERMAL_STATE" != nominal ]]; then
      echo "calibrated phase $phase observed non-nominal thermal state: $THERMAL_STATE" >&2
      return 1
    fi
    sleep "$sample_seconds"
  done
}

thermal_monitor_fair_or_better_while_pid() {
  local log_file=$1
  local phase=$2
  local producer_pid=$3
  local sample_seconds=$4

  [[ "$producer_pid" =~ ^[1-9][0-9]*$ ]] || {
    echo "thermal producer pid must be a positive integer" >&2
    return 2
  }
  [[ "$sample_seconds" =~ ^[0-9]+$ ]] || {
    echo "thermal sample interval must be a non-negative integer" >&2
    return 2
  }
  while :; do
    thermal_read_process_state "$producer_pid" || return 1
    if [[ -z "$THERMAL_PROCESS_STATE" || "$THERMAL_PROCESS_STATE" == Z* ]]; then
      return 0
    fi
    thermal_sample "$log_file" "$phase" || return 1
    case "$THERMAL_STATE" in
      nominal|fair) ;;
      serious|critical)
        echo "calibrated phase $phase exceeded fair thermal state: $THERMAL_STATE" >&2
        return 1
        ;;
      *)
        echo "calibrated phase $phase observed invalid thermal state: $THERMAL_STATE" >&2
        return 1
        ;;
    esac
    sleep "$sample_seconds"
  done
}

thermal_prepare_cold_receipt_dir() {
  local receipt_dir=$1
  local existing_receipt

  mkdir -p "$receipt_dir" || return 1
  existing_receipt=$(find "$receipt_dir" -maxdepth 1 -type f \
    -name 'agent-*.cold.json' -print -quit) || return 1
  if [[ -n "$existing_receipt" ]]; then
    echo "cold-cohort receipt directory contains stale evidence: $existing_receipt" >&2
    return 1
  fi
}

thermal_monitor_nominal_until_cold_receipts() {
  local log_file=$1
  local phase=$2
  local receipt_dir=$3
  local expected_receipts=$4
  local sample_seconds=$5
  local timeout_seconds=$6
  local producer_pid=${7:-}
  local deadline
  local producer_state
  local receipt_index
  local receipt_count
  local receipt_set_complete

  for value in "$expected_receipts" "$sample_seconds" "$timeout_seconds"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
      echo "cold-cohort thermal values must be non-negative integers" >&2
      return 2
    }
  done
  ((expected_receipts > 0)) || {
    echo "cold-cohort thermal monitor requires at least one receipt" >&2
    return 2
  }
  [[ -d "$receipt_dir" ]] || {
    echo "cold-cohort receipt directory is missing: $receipt_dir" >&2
    return 1
  }

  deadline=$((SECONDS + timeout_seconds))
  while :; do
    if [[ -n "$producer_pid" ]]; then
      producer_state=$(ps -p "$producer_pid" -o state= 2>/dev/null \
        | tr -d '[:space:]')
      if [[ -z "$producer_state" || "$producer_state" == Z* ]]; then
        echo "cold-cohort producer exited before publishing all receipts" >&2
        return 1
      fi
    fi
    thermal_sample "$log_file" "$phase" || return 1
    if [[ "$THERMAL_STATE" != nominal ]]; then
      echo "calibrated phase $phase observed non-nominal thermal state: $THERMAL_STATE" >&2
      return 1
    fi
    receipt_count=$(find "$receipt_dir" -maxdepth 1 -type f \
      -name 'agent-*.cold.json' -size +0c | wc -l | tr -d '[:space:]')
    [[ "$receipt_count" =~ ^[0-9]+$ ]] || {
      echo "failed to count cold-cohort receipts" >&2
      return 1
    }
    receipt_set_complete=1
    for ((receipt_index = 1; receipt_index <= expected_receipts; receipt_index++)); do
      if [[ ! -s "$receipt_dir/agent-${receipt_index}.cold.json" ]]; then
        receipt_set_complete=0
        break
      fi
    done
    if ((receipt_count == expected_receipts && receipt_set_complete == 0)); then
      echo "cold-cohort receipts do not match agent-1 through agent-${expected_receipts}" >&2
      return 1
    fi
    if ((receipt_count == expected_receipts && receipt_set_complete == 1)); then
      thermal_sample "$log_file" "$phase-end" || return 1
      [[ "$THERMAL_STATE" == nominal ]] || {
        echo "calibrated phase $phase ended in non-nominal thermal state: $THERMAL_STATE" >&2
        return 1
      }
      return 0
    fi
    if ((receipt_count > expected_receipts)); then
      echo "cold-cohort receipt count exceeded ${expected_receipts}: $receipt_count" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "cold cohort did not publish ${expected_receipts} receipts within ${timeout_seconds}s" >&2
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
  # These globals are the validator's output contract for release callers.
  # shellcheck disable=SC2034
  IFS=$'\t' read -r THERMAL_LOG_SAMPLES THERMAL_LOG_DURATION_SECONDS \
    THERMAL_LOG_NON_NOMINAL_SAMPLES THERMAL_LOG_GAPS THERMAL_LOG_INVALID_ROWS \
    <<<"$stats"
  ((THERMAL_LOG_SAMPLES >= 2)) \
    && ((THERMAL_LOG_DURATION_SECONDS > 0)) \
    && ((THERMAL_LOG_NON_NOMINAL_SAMPLES == 0)) \
    && ((THERMAL_LOG_GAPS == 0)) \
    && ((THERMAL_LOG_INVALID_ROWS == 0))
}

thermal_validate_fair_or_better_measurement_log() {
  local log_file=$1
  local maximum_gap_seconds=$2
  local stats

  [[ "$maximum_gap_seconds" =~ ^[0-9]+$ ]] || return 2
  stats=$(awk -F '\t' -v maximum="$maximum_gap_seconds" '
    BEGIN { invalid = 0; gaps = 0; non_nominal = 0; fair = 0; over_limit = 0 }
    {
      if (NF != 3 || $1 !~ /^[0-9]+$/ \
          || $2 !~ /^(nominal|fair|serious|critical)$/) {
        invalid++
        next
      }
      samples++
      if ($2 != "nominal") non_nominal++
      if ($2 == "fair") fair++
      if ($2 == "serious" || $2 == "critical") over_limit++
      if (samples == 1) first = $1
      if (samples > 1 && ($1 < previous || $1 - previous > maximum)) gaps++
      previous = $1
      last = $1
    }
    END {
      duration = samples > 0 ? last - first : -1
      printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\n", samples, duration,
        non_nominal, fair, over_limit, gaps, invalid
    }
  ' "$log_file") || return 1
  # These globals are the validator's output contract for release callers.
  # shellcheck disable=SC2034
  IFS=$'\t' read -r THERMAL_LOG_SAMPLES THERMAL_LOG_DURATION_SECONDS \
    THERMAL_LOG_NON_NOMINAL_SAMPLES THERMAL_LOG_FAIR_SAMPLES \
    THERMAL_LOG_OVER_LIMIT_SAMPLES THERMAL_LOG_GAPS \
    THERMAL_LOG_INVALID_ROWS <<<"$stats"
  ((THERMAL_LOG_SAMPLES >= 2)) \
    && ((THERMAL_LOG_DURATION_SECONDS > 0)) \
    && ((THERMAL_LOG_OVER_LIMIT_SAMPLES == 0)) \
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
