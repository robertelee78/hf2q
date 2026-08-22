#!/usr/bin/env bash
# Source-only macOS VM telemetry for protected hardware gates.
# shellcheck disable=SC2034 # Public globals are consumed by sourcing runners.

MEMORY_PRESSURE_POLICY="darwin25-phase-bound-no-vm-churn-v2"
MEMORY_PRESSURE_NORMAL_LEVEL=1
MEMORY_PRESSURE_WARNING_LEVEL=2
MEMORY_PRESSURE_CRITICAL_LEVEL=4

memory_read_sample() {
  local vm_output pressure_output boot_output parsed

  MEMORY_PRESSURE_LEVEL=$(/usr/sbin/sysctl -n \
    kern.memorystatus_vm_pressure_level 2>/dev/null) || {
    echo "failed to read macOS kernel memory-pressure level" >&2
    return 1
  }
  boot_output=$(/usr/sbin/sysctl -n kern.boottime 2>/dev/null) || {
    echo "failed to read macOS boot epoch" >&2
    return 1
  }
  MEMORY_BOOT_TIME_SECONDS=$(awk '
    match($0, /sec = [0-9]+/) {
      value=substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", value)
      print value; found=1; exit
    }
    END { if (!found) exit 1 }
  ' <<<"$boot_output") || {
    echo "macOS kern.boottime was malformed" >&2
    return 1
  }
  vm_output=$(/usr/bin/vm_stat 2>/dev/null) || {
    echo "failed to read macOS VM counters" >&2
    return 1
  }
  parsed=$(awk -F: '
    NR == 1 && match($0, /page size of [0-9]+ bytes/) {
      value=substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", value)
      page_size=value; found["page_size"]=1
    }
    function number(label, key, value) {
      if ($1 == label) {
        value=$2; gsub(/[^0-9]/, "", value)
        values[key]=value; found[key]=1
      }
    }
    {
      number("Pageins", "pageins")
      number("Pageouts", "pageouts")
      number("Swapins", "swapins")
      number("Swapouts", "swapouts")
      number("Compressions", "compressions")
      number("Decompressions", "decompressions")
      number("Pages purged", "purges")
      number("Pages reactivated", "reactivations")
      number("Pages throttled", "throttled")
      number("Pages wired down", "wired")
      number("Pages occupied by compressor", "compressor")
      number("Pages stored in compressor", "uncompressed_compressor")
    }
    END {
      required[1]="page_size"; required[2]="pageins"; required[3]="pageouts"
      required[4]="swapins"; required[5]="swapouts"
      required[6]="compressions"; required[7]="decompressions"
      required[8]="purges"; required[9]="reactivations"
      required[10]="throttled"; required[11]="wired"
      required[12]="compressor"; required[13]="uncompressed_compressor"
      for (i=1; i<=13; i++) if (!found[required[i]]) exit 1
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
        page_size, values["pageins"], values["pageouts"], values["swapins"],
        values["swapouts"], values["compressions"], values["decompressions"],
        values["purges"], values["reactivations"], values["throttled"],
        values["wired"], values["compressor"], values["uncompressed_compressor"]
    }
  ' <<<"$vm_output") || {
    echo "macOS vm_stat omitted a required VM field" >&2
    return 1
  }
  IFS=$'\t' read -r MEMORY_PAGE_SIZE MEMORY_PAGEINS MEMORY_PAGEOUTS \
    MEMORY_SWAPINS MEMORY_SWAPOUTS MEMORY_COMPRESSIONS MEMORY_DECOMPRESSIONS \
    MEMORY_PURGES MEMORY_REACTIVATIONS MEMORY_THROTTLED_PAGES \
    MEMORY_WIRED_PAGES MEMORY_COMPRESSOR_PAGES \
    MEMORY_UNCOMPRESSED_COMPRESSOR_PAGES <<<"$parsed"
  pressure_output=$(/usr/bin/memory_pressure -Q 2>/dev/null) || {
    echo "failed to read macOS memory free percentage" >&2
    return 1
  }
  MEMORY_FREE_PERCENTAGE=$(awk -F: \
    '/^System-wide memory free percentage:/ {
      gsub(/[^0-9]/, "", $2); print $2; found=1; exit
    } END { if (!found) exit 1 }' <<<"$pressure_output") || {
    echo "memory_pressure did not report system-wide free percentage" >&2
    return 1
  }
  for value in "$MEMORY_BOOT_TIME_SECONDS" "$MEMORY_PAGE_SIZE" \
    "$MEMORY_PRESSURE_LEVEL" "$MEMORY_FREE_PERCENTAGE" "$MEMORY_PAGEINS" \
    "$MEMORY_PAGEOUTS" "$MEMORY_SWAPINS" "$MEMORY_SWAPOUTS" \
    "$MEMORY_COMPRESSIONS" "$MEMORY_DECOMPRESSIONS" "$MEMORY_PURGES" \
    "$MEMORY_REACTIVATIONS" "$MEMORY_THROTTLED_PAGES" \
    "$MEMORY_WIRED_PAGES" "$MEMORY_COMPRESSOR_PAGES" \
    "$MEMORY_UNCOMPRESSED_COMPRESSOR_PAGES"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
      echo "malformed macOS memory sample value: $value" >&2
      return 1
    }
  done
  ((MEMORY_FREE_PERCENTAGE <= 100 && MEMORY_PAGE_SIZE > 0)) || return 1
}

memory_capture_line() {
  local phase=$1

  memory_read_sample || return 1
  MEMORY_SAMPLED_AT=$(/bin/date +%s) || return 1
  [[ "$MEMORY_SAMPLED_AT" =~ ^[0-9]+$ && -n "$phase" ]] || return 1
  printf -v MEMORY_SAMPLE_LINE \
    '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "$MEMORY_SAMPLED_AT" "$MEMORY_BOOT_TIME_SECONDS" "$MEMORY_PAGE_SIZE" \
    "$MEMORY_PRESSURE_LEVEL" "$MEMORY_FREE_PERCENTAGE" "$MEMORY_PAGEINS" \
    "$MEMORY_PAGEOUTS" "$MEMORY_SWAPINS" "$MEMORY_SWAPOUTS" \
    "$MEMORY_COMPRESSIONS" "$MEMORY_DECOMPRESSIONS" "$MEMORY_PURGES" \
    "$MEMORY_REACTIVATIONS" "$MEMORY_THROTTLED_PAGES" \
    "$MEMORY_WIRED_PAGES" "$MEMORY_COMPRESSOR_PAGES" \
    "$MEMORY_UNCOMPRESSED_COMPRESSOR_PAGES" "$phase"
}

memory_require_warning_or_better() {
  local phase=$1
  case "$MEMORY_PRESSURE_LEVEL" in
    "$MEMORY_PRESSURE_NORMAL_LEVEL"|"$MEMORY_PRESSURE_WARNING_LEVEL") ;;
    *)
      echo "calibrated phase $phase exceeded warning memory pressure: $MEMORY_PRESSURE_LEVEL" >&2
      return 1
      ;;
  esac
  if ((MEMORY_THROTTLED_PAGES != 0)); then
    echo "calibrated phase $phase observed throttled VM pages: $MEMORY_THROTTLED_PAGES" >&2
    return 1
  fi
}

memory_sample() {
  local log_file=$1
  local phase=$2
  local expected_setup_swapouts=${3:-}

  memory_capture_line "$phase" || return 1
  printf '%s\n' "$MEMORY_SAMPLE_LINE" >>"$log_file"
  memory_require_warning_or_better "$phase" || return 1
  if [[ -n "$expected_setup_swapouts" \
    && "$MEMORY_SWAPOUTS" != "$expected_setup_swapouts" ]]; then
    echo "setup phase $phase observed swapout growth: $expected_setup_swapouts -> $MEMORY_SWAPOUTS" >&2
    return 1
  fi
}

memory_validate_warning_log() {
  local log_file=$1
  local maximum_gap_seconds=$2
  local require_flat_swapouts=${3:-0}
  local stats

  [[ -s "$log_file" ]] || {
    echo "memory-pressure log is missing or empty: $log_file" >&2
    return 1
  }
  [[ "$maximum_gap_seconds" =~ ^[1-9][0-9]*$ \
    && "$require_flat_swapouts" =~ ^[01]$ ]] || return 2
  stats=$(awk -F '\t' -v max_gap="$maximum_gap_seconds" \
    -v flat_swap="$require_flat_swapouts" '
    NF != 18 { invalid=1; next }
    {
      row_invalid=0
      for (i=1; i<=17; i++) if ($i !~ /^[0-9]+$/) row_invalid=1
      if ($18 == "" || ($4 != 1 && $4 != 2) || $5 > 100 || $14 != 0)
        row_invalid=1
      if (row_invalid) { invalid=1; next }
      if (NR == 1) {
        first_time=$1; boot=$2; page_size=$3; min_free=$5; max_level=$4
        first_pageins=$6; first_pageouts=$7; first_swapins=$8; first_swapouts=$9
        first_compressions=$10; first_decompressions=$11; first_purges=$12
        first_reactivations=$13
      } else {
        if ($1 < previous_time || $1-previous_time > max_gap ||
            $2 != boot || $3 != page_size || $6 < previous_pageins ||
            $7 < previous_pageouts || $8 < previous_swapins ||
            $9 < previous_swapouts || $10 < previous_compressions ||
            $11 < previous_decompressions || $12 < previous_purges ||
            $13 < previous_reactivations) { invalid=1; next }
      }
      if (flat_swap && $9 != first_swapouts) { invalid=1; next }
      if ($4 == 1) normal++; else warning++
      if ($5 < min_free) min_free=$5
      if ($4 > max_level) max_level=$4
      previous_time=$1; previous_pageins=$6; previous_pageouts=$7
      previous_swapins=$8; previous_swapouts=$9; previous_compressions=$10
      previous_decompressions=$11; previous_purges=$12; previous_reactivations=$13
      last_time=$1; last_pageins=$6; last_pageouts=$7; last_swapins=$8
      last_swapouts=$9; last_compressions=$10; last_decompressions=$11
      last_purges=$12; last_reactivations=$13; samples++
    }
    END {
      if (invalid || samples < 2) exit 1
      printf "%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s", samples,
        last_time-first_time, normal, warning, min_free, max_level, boot, page_size
      printf "\t%s\t%s\t%d", first_pageins, last_pageins,
        last_pageins-first_pageins
      printf "\t%s\t%s\t%d", first_pageouts, last_pageouts,
        last_pageouts-first_pageouts
      printf "\t%s\t%s\t%d", first_swapins, last_swapins,
        last_swapins-first_swapins
      printf "\t%s\t%s\t%d", first_swapouts, last_swapouts,
        last_swapouts-first_swapouts
      printf "\t%s\t%s\t%d", first_compressions, last_compressions,
        last_compressions-first_compressions
      printf "\t%s\t%s\t%d", first_decompressions, last_decompressions,
        last_decompressions-first_decompressions
      printf "\t%s\t%s\t%d", first_purges, last_purges,
        last_purges-first_purges
      printf "\t%s\t%s\t%d\n", first_reactivations, last_reactivations,
        last_reactivations-first_reactivations
    }
  ' "$log_file") || {
    echo "memory-pressure log violates warning-or-better telemetry contract" >&2
    return 1
  }
  IFS=$'\t' read -r MEMORY_LOG_SAMPLES MEMORY_LOG_DURATION_SECONDS \
    MEMORY_LOG_NORMAL_SAMPLES MEMORY_LOG_WARNING_SAMPLES \
    MEMORY_LOG_MIN_FREE_PERCENTAGE MEMORY_LOG_MAX_PRESSURE_LEVEL \
    MEMORY_LOG_BOOT_TIME_SECONDS MEMORY_LOG_PAGE_SIZE \
    MEMORY_LOG_INITIAL_PAGEINS MEMORY_LOG_FINAL_PAGEINS MEMORY_LOG_PAGEIN_DELTA \
    MEMORY_LOG_INITIAL_PAGEOUTS MEMORY_LOG_FINAL_PAGEOUTS MEMORY_LOG_PAGEOUT_DELTA \
    MEMORY_LOG_INITIAL_SWAPINS MEMORY_LOG_FINAL_SWAPINS MEMORY_LOG_SWAPIN_DELTA \
    MEMORY_LOG_INITIAL_SWAPOUTS MEMORY_LOG_FINAL_SWAPOUTS MEMORY_LOG_SWAPOUT_DELTA \
    MEMORY_LOG_INITIAL_COMPRESSIONS MEMORY_LOG_FINAL_COMPRESSIONS \
    MEMORY_LOG_COMPRESSION_DELTA MEMORY_LOG_INITIAL_DECOMPRESSIONS \
    MEMORY_LOG_FINAL_DECOMPRESSIONS MEMORY_LOG_DECOMPRESSION_DELTA \
    MEMORY_LOG_INITIAL_PURGES MEMORY_LOG_FINAL_PURGES MEMORY_LOG_PURGE_DELTA \
    MEMORY_LOG_INITIAL_REACTIVATIONS MEMORY_LOG_FINAL_REACTIVATIONS \
    MEMORY_LOG_REACTIVATION_DELTA <<<"$stats"
  MEMORY_LOG_MAX_THROTTLED_PAGES=0
}

memory_log_summary_json() {
  jq -n \
    --argjson samples "$MEMORY_LOG_SAMPLES" \
    --argjson duration "$MEMORY_LOG_DURATION_SECONDS" \
    --argjson normal "$MEMORY_LOG_NORMAL_SAMPLES" \
    --argjson warning "$MEMORY_LOG_WARNING_SAMPLES" \
    --argjson min_free "$MEMORY_LOG_MIN_FREE_PERCENTAGE" \
    --argjson max_pressure "$MEMORY_LOG_MAX_PRESSURE_LEVEL" \
    --argjson boot "$MEMORY_LOG_BOOT_TIME_SECONDS" \
    --argjson page_size "$MEMORY_LOG_PAGE_SIZE" \
    --argjson pageins "$MEMORY_LOG_PAGEIN_DELTA" \
    --argjson pageouts "$MEMORY_LOG_PAGEOUT_DELTA" \
    --argjson swapins "$MEMORY_LOG_SWAPIN_DELTA" \
    --argjson swapouts "$MEMORY_LOG_SWAPOUT_DELTA" \
    --argjson compressions "$MEMORY_LOG_COMPRESSION_DELTA" \
    --argjson decompressions "$MEMORY_LOG_DECOMPRESSION_DELTA" \
    --argjson purges "$MEMORY_LOG_PURGE_DELTA" \
    --argjson reactivations "$MEMORY_LOG_REACTIVATION_DELTA" '
      {samples:$samples,duration_seconds:$duration,
       normal_samples:$normal,warning_samples:$warning,
       min_free_percentage:$min_free,max_pressure_level:$max_pressure,
       max_throttled_pages:0,boot_time_seconds:$boot,page_size:$page_size,
       observed_deltas:{pageins:$pageins,pageouts:$pageouts,
         swapins:$swapins,swapouts:$swapouts,compressions:$compressions,
         decompressions:$decompressions,purges:$purges,
         reactivations:$reactivations}}'
}
