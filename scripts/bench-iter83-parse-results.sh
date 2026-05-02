#!/bin/bash
# iter83 result parser — extracts prefill ms / tok/s from the alternating
# A/B logs and computes the trimmed median + min/max + IQR for each
# binary on each workload.
#
# Usage:
#   /tmp/cfa-iter83/impl-bench/parse-results.sh <workload>
#
# <workload> ∈ { chunk-engaged | default }
set -euo pipefail

WORKLOAD="${1:-chunk-engaged}"
LOGDIR="/tmp/cfa-iter83/impl-bench/logs/${WORKLOAD}"

if [ ! -d "${LOGDIR}" ]; then
    echo "ERROR: log dir not found: ${LOGDIR}"
    exit 1
fi

extract_ms() {
    grep "^prefill:" "$1" 2>/dev/null | tail -1 | sed -E 's/.* in ([0-9]+)ms.*/\1/'
}

echo "=== iter83 ${WORKLOAD} bench results ==="
echo

# Trial-level table
printf "%-20s %10s\n" "label" "prefill_ms"
printf "%-20s %10s\n" "--------" "----------"
for label in warmup-iter83 warmup-iter82 \
             A1-iter83 B1-iter82 \
             A2-iter83 B2-iter82 \
             A3-iter83 B3-iter82; do
    log="${LOGDIR}/${label}.log"
    if [ -f "${log}" ]; then
        ms=$(extract_ms "${log}")
        printf "%-20s %10s\n" "${label}" "${ms:-???}"
    else
        printf "%-20s %10s\n" "${label}" "(missing)"
    fi
done

echo

# Per-binary summary (excluding warmup): trimmed median, min, max
summarize() {
    local binlabel="$1"
    local logs=("${LOGDIR}/A1-${binlabel}.log" "${LOGDIR}/A2-${binlabel}.log" "${LOGDIR}/A3-${binlabel}.log")
    if [ "${binlabel}" == "iter82" ]; then
        logs=("${LOGDIR}/B1-${binlabel}.log" "${LOGDIR}/B2-${binlabel}.log" "${LOGDIR}/B3-${binlabel}.log")
    fi
    local vals=()
    for log in "${logs[@]}"; do
        if [ -f "${log}" ]; then
            local v=$(extract_ms "${log}")
            if [ -n "${v}" ]; then
                vals+=("${v}")
            fi
        fi
    done
    if [ "${#vals[@]}" -eq 0 ]; then
        printf "%-12s no data\n" "${binlabel}"
        return
    fi
    # Sort
    IFS=$'\n' sorted=($(sort -n <<<"${vals[*]}"))
    unset IFS
    local n=${#sorted[@]}
    local median
    if [ "$n" -eq 3 ]; then
        median="${sorted[1]}"
    elif [ "$n" -eq 2 ]; then
        median=$(awk "BEGIN { printf \"%.0f\", (${sorted[0]} + ${sorted[1]}) / 2 }")
    else
        median="${sorted[0]}"
    fi
    local mn="${sorted[0]}"
    local mx="${sorted[$((n - 1))]}"
    local iqr=$((mx - mn))
    printf "%-12s n=%d median=%sms min=%sms max=%sms IQR=%sms\n" \
        "${binlabel}" "$n" "${median}" "${mn}" "${mx}" "${iqr}"
}

summarize iter83
summarize iter82

echo
echo "=== A/B delta (iter83 - iter82, both medians) ==="
get_median() {
    local binlabel="$1"
    local logs=()
    if [ "${binlabel}" == "iter83" ]; then
        logs=("${LOGDIR}/A1-iter83.log" "${LOGDIR}/A2-iter83.log" "${LOGDIR}/A3-iter83.log")
    else
        logs=("${LOGDIR}/B1-iter82.log" "${LOGDIR}/B2-iter82.log" "${LOGDIR}/B3-iter82.log")
    fi
    local vals=()
    for log in "${logs[@]}"; do
        local v=$(extract_ms "${log}" 2>/dev/null)
        [ -n "${v}" ] && vals+=("${v}")
    done
    if [ "${#vals[@]}" -ne 3 ]; then return; fi
    IFS=$'\n' sorted=($(sort -n <<<"${vals[*]}"))
    unset IFS
    echo "${sorted[1]}"
}

m83=$(get_median iter83)
m82=$(get_median iter82)
if [ -n "${m83}" ] && [ -n "${m82}" ]; then
    delta=$((m83 - m82))
    sign="+"
    [ "${delta}" -lt 0 ] && sign=""
    pct=$(awk "BEGIN { printf \"%.2f\", 100 * (${m83} - ${m82}) / ${m82} }")
    echo "iter83 median=${m83}ms"
    echo "iter82 median=${m82}ms"
    echo "delta=${sign}${delta}ms (${pct}%)"
    if [ "${delta}" -lt -50 ]; then
        echo "VERDICT: WIN (>= 50ms improvement)"
    elif [ "${delta}" -lt -30 ]; then
        echo "VERDICT: PARTIAL (>= 30ms improvement, < 50ms)"
    elif [ "${delta}" -lt 30 ]; then
        echo "VERDICT: NEUTRAL (within ±30ms noise floor)"
    else
        echo "VERDICT: REGRESSION (>= 30ms slower)"
    fi
else
    echo "incomplete data"
fi
