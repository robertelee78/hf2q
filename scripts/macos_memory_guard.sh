#!/usr/bin/env bash
# Source-only macOS memory-pressure and swap guard for protected hardware gates.
# shellcheck disable=SC2034 # Public globals are consumed by sourcing runners.

MEMORY_PRESSURE_POLICY="darwin25-normal-no-swapout-v1"
MEMORY_PRESSURE_NORMAL_LEVEL=1

memory_read_sample() {
  local vm_output
  local pressure_output

  MEMORY_PRESSURE_LEVEL=$(/usr/sbin/sysctl -n \
    kern.memorystatus_vm_pressure_level 2>/dev/null) || {
    echo "failed to read macOS kernel memory-pressure level" >&2
    return 1
  }
  vm_output=$(/usr/bin/vm_stat 2>/dev/null) || {
    echo "failed to read macOS VM counters" >&2
    return 1
  }
  pressure_output=$(/usr/bin/memory_pressure -Q 2>/dev/null) || {
    echo "failed to read macOS memory free percentage" >&2
    return 1
  }
  MEMORY_SWAPOUTS=$(awk -F: '/^Swapouts:/ {
    gsub(/[^0-9]/, "", $2); print $2; found=1; exit
  } END { if (!found) exit 1 }' <<<"$vm_output") || {
    echo "macOS vm_stat did not report Swapouts" >&2
    return 1
  }
  MEMORY_THROTTLED_PAGES=$(awk -F: '/^Pages throttled:/ {
    gsub(/[^0-9]/, "", $2); print $2; found=1; exit
  } END { if (!found) exit 1 }' <<<"$vm_output") || {
    echo "macOS vm_stat did not report Pages throttled" >&2
    return 1
  }
  MEMORY_FREE_PERCENTAGE=$(awk -F: \
    '/^System-wide memory free percentage:/ {
      gsub(/[^0-9]/, "", $2); print $2; found=1; exit
    } END { if (!found) exit 1 }' <<<"$pressure_output") || {
    echo "memory_pressure did not report system-wide free percentage" >&2
    return 1
  }
  for value in "$MEMORY_PRESSURE_LEVEL" "$MEMORY_SWAPOUTS" \
    "$MEMORY_THROTTLED_PAGES" "$MEMORY_FREE_PERCENTAGE"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
      echo "malformed macOS memory sample value: $value" >&2
      return 1
    }
  done
  ((MEMORY_FREE_PERCENTAGE <= 100)) || {
    echo "macOS memory free percentage exceeds 100: $MEMORY_FREE_PERCENTAGE" >&2
    return 1
  }
}

memory_sample() {
  local log_file=$1
  local phase=$2
  local expected_swapouts=${3:-}
  local sampled_at

  memory_read_sample || return 1
  sampled_at=$(/bin/date +%s) || return 1
  [[ "$sampled_at" =~ ^[0-9]+$ ]] || return 1
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$sampled_at" "$MEMORY_PRESSURE_LEVEL" "$MEMORY_SWAPOUTS" \
    "$MEMORY_THROTTLED_PAGES" "$MEMORY_FREE_PERCENTAGE" "$phase" \
    >>"$log_file"
  if [[ "$MEMORY_PRESSURE_LEVEL" != "$MEMORY_PRESSURE_NORMAL_LEVEL" ]]; then
    echo "calibrated phase $phase observed non-normal memory pressure: $MEMORY_PRESSURE_LEVEL" >&2
    return 1
  fi
  if [[ -n "$expected_swapouts" && "$MEMORY_SWAPOUTS" != "$expected_swapouts" ]]; then
    echo "calibrated phase $phase observed swapout growth: $expected_swapouts -> $MEMORY_SWAPOUTS" >&2
    return 1
  fi
  if ((MEMORY_THROTTLED_PAGES != 0)); then
    echo "calibrated phase $phase observed throttled VM pages: $MEMORY_THROTTLED_PAGES" >&2
    return 1
  fi
}

memory_monitor_normal_no_swapout_while_pid() {
  local log_file=$1
  local phase=$2
  local producer_pid=$3
  local sample_seconds=$4
  local expected_swapouts=$5
  local producer_state

  [[ "$producer_pid" =~ ^[1-9][0-9]*$ ]] || return 2
  [[ "$sample_seconds" =~ ^[1-9][0-9]*$ ]] || return 2
  [[ "$expected_swapouts" =~ ^[0-9]+$ ]] || return 2
  while :; do
    producer_state=$(/bin/ps -p "$producer_pid" -o state= 2>/dev/null \
      | /usr/bin/tr -d '[:space:]')
    if [[ -z "$producer_state" || "$producer_state" == Z* ]]; then
      return 0
    fi
    memory_sample "$log_file" "$phase" "$expected_swapouts" || return 1
    /bin/sleep "$sample_seconds"
  done
}

memory_validate_normal_no_swapout_log() {
  local log_file=$1
  local maximum_gap_seconds=$2
  local stats

  [[ -s "$log_file" ]] || {
    echo "memory-pressure log is missing or empty: $log_file" >&2
    return 1
  }
  [[ "$maximum_gap_seconds" =~ ^[1-9][0-9]*$ ]] || return 2
  stats=$(awk -F '\t' \
    -v normal="$MEMORY_PRESSURE_NORMAL_LEVEL" \
    -v max_gap="$maximum_gap_seconds" '
    NF != 6 { exit 1 }
    $1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/ ||
      $4 !~ /^[0-9]+$/ || $5 !~ /^[0-9]+$/ || $6 == "" { exit 1 }
    $2 != normal || $4 != 0 || $5 > 100 { exit 1 }
    NR == 1 { first_time=$1; first_swap=$3; min_free=$5; max_level=$2;
      max_throttled=$4 }
    NR > 1 && ($1 < previous_time || $1 - previous_time > max_gap) { exit 1 }
    $3 != first_swap { exit 1 }
    $5 < min_free { min_free=$5 }
    $2 > max_level { max_level=$2 }
    $4 > max_throttled { max_throttled=$4 }
    { previous_time=$1; last_time=$1; last_swap=$3; samples++ }
    END {
      if (samples < 2) exit 1
      printf "%d\t%d\t%s\t%s\t%d\t%d\n", samples,
        last_time-first_time, first_swap, last_swap, min_free, max_level
    }
  ' "$log_file") || {
    echo "memory-pressure log violates normal/no-swapout contract" >&2
    return 1
  }
  IFS=$'\t' read -r MEMORY_LOG_SAMPLES MEMORY_LOG_DURATION_SECONDS \
    MEMORY_LOG_INITIAL_SWAPOUTS MEMORY_LOG_FINAL_SWAPOUTS \
    MEMORY_LOG_MIN_FREE_PERCENTAGE MEMORY_LOG_MAX_PRESSURE_LEVEL <<<"$stats"
  MEMORY_LOG_SWAPOUT_DELTA=$((MEMORY_LOG_FINAL_SWAPOUTS - MEMORY_LOG_INITIAL_SWAPOUTS))
  MEMORY_LOG_MAX_THROTTLED_PAGES=0
}

memory_validate_measurement_coverage() {
  local memory_log=$1
  local measurement_log=$2
  local maximum_gap_seconds=$3
  local memory_bounds measurement_bounds
  local memory_first memory_last measurement_first measurement_last
  local start_delta end_delta duration_delta

  [[ -s "$memory_log" && -s "$measurement_log" ]] || {
    echo "memory or measurement log is missing for coverage validation" >&2
    return 1
  }
  [[ "$maximum_gap_seconds" =~ ^[1-9][0-9]*$ ]] || return 2
  memory_bounds=$(awk -F '\t' '
    NF != 6 { exit 1 }
    NR == 1 && $6 != "decode-cohort-measurement-start" { exit 1 }
    NR > 1 && $6 != "decode-cohort-measurement" &&
      $6 != "decode-cohort-measurement-end" { exit 1 }
    $6 == "decode-cohort-measurement-start" { starts++ }
    $6 == "decode-cohort-measurement-end" { ends++ }
    NR == 1 { first=$1 }
    { last=$1; last_phase=$6 }
    END {
      if (starts != 1 || ends != 1 ||
          last_phase != "decode-cohort-measurement-end") exit 1
      printf "%s\t%s\n", first, last
    }
  ' "$memory_log") || {
    echo "memory-pressure log does not span the measurement phases" >&2
    return 1
  }
  measurement_bounds=$(awk -F '\t' '
    NF != 3 { exit 1 }
    NR == 1 && $3 != "decode-cohort-measurement-start" { exit 1 }
    NR == 1 { first=$1 }
    { last=$1; last_phase=$3 }
    END {
      if (last_phase != "decode-cohort-measurement-end") exit 1
      printf "%s\t%s\n", first, last
    }
  ' "$measurement_log") || {
    echo "thermal log does not span the measurement phases" >&2
    return 1
  }
  IFS=$'\t' read -r memory_first memory_last <<<"$memory_bounds"
  IFS=$'\t' read -r measurement_first measurement_last \
    <<<"$measurement_bounds"
  for value in "$memory_first" "$memory_last" "$measurement_first" \
    "$measurement_last"; do
    [[ "$value" =~ ^[0-9]+$ ]] || return 1
  done
  start_delta=$((memory_first - measurement_first))
  end_delta=$((memory_last - measurement_last))
  duration_delta=$(((memory_last - memory_first) - \
    (measurement_last - measurement_first)))
  ((start_delta < 0)) && start_delta=$((-start_delta))
  ((end_delta < 0)) && end_delta=$((-end_delta))
  ((duration_delta < 0)) && duration_delta=$((-duration_delta))
  if ((start_delta > maximum_gap_seconds ||
      end_delta > maximum_gap_seconds ||
      duration_delta > maximum_gap_seconds)); then
    echo "memory-pressure log does not cover the thermal measurement window" >&2
    return 1
  fi
}
