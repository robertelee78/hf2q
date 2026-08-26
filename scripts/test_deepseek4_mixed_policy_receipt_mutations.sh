#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt=${1:?usage: test_deepseek4_mixed_policy_receipt_mutations.sh RECEIPT SOURCE_ROOT}
source_root=${2:?usage: test_deepseek4_mixed_policy_receipt_mutations.sh RECEIPT SOURCE_ROOT}
[[ "$receipt" == /* && -f "$receipt" && "$source_root" == /* ]] || {
    echo "DeepSeek B.1 mutations require absolute receipt and source paths" >&2
    exit 2
}
evidence_root=$(cd "$(dirname "$receipt")" && pwd -P)
tmp_dir=$(mktemp -d "${TMPDIR:-/var/tmp}/deepseek-b1-mutations.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

verify() {
    python3 "$script_dir/verify_deepseek4_mixed_policy_receipt.py" "$1" "$source_root"
}

verify "$receipt" >/dev/null

copy_evidence() {
    local name=$1
    local destination="$tmp_dir/$name"
    cp -R "$evidence_root" "$destination"
    printf '%s\n' "$destination"
}

expect_reject() {
    local name=$1 copy
    copy=$(copy_evidence "$name")
    shift
    "$@" "$copy"
    if verify "$copy/receipt.json" >/dev/null 2>&1; then
        echo "receipt verifier accepted injected mutation: $name" >&2
        exit 1
    fi
}

mutate_top() {
    local filter=$1 copy=$2
    jq "$filter" "$copy/receipt.json" >"$copy/receipt.json.tmp"
    mv "$copy/receipt.json.tmp" "$copy/receipt.json"
}

seal_process() {
    local copy=$1 label=$2 manifest_sha summary_sha
    local process_dir="$copy/$label"
    (
      cd "$process_dir"
      find . -type f ! -name evidence.sha256 ! -name summary.json \
        | sed 's#^./##' | sort | while IFS= read -r relative; do
          printf '%s  %s\n' "$(shasum -a 256 "$relative" | awk '{print $1}')" "$relative"
        done >evidence.sha256.tmp
      mv evidence.sha256.tmp evidence.sha256
    )
    manifest_sha=$(shasum -a 256 "$process_dir/evidence.sha256" | awk '{print $1}')
    jq --arg sha "$manifest_sha" '.evidence_manifest_sha256=$sha' \
      "$process_dir/summary.json" >"$process_dir/summary.json.tmp"
    mv "$process_dir/summary.json.tmp" "$process_dir/summary.json"
    summary_sha=$(shasum -a 256 "$process_dir/summary.json" | awk '{print $1}')
    jq --arg label "$label" --arg manifest "$manifest_sha" --arg summary "$summary_sha" '
      .evidence.processes[$label].manifest_sha256=$manifest
      | .evidence.processes[$label].summary_sha256=$summary
    ' "$copy/receipt.json" >"$copy/receipt.json.tmp"
    mv "$copy/receipt.json.tmp" "$copy/receipt.json"
}

mutate_timing() {
    local copy=$1
    printf '1000.000000000\t999.000000000\n' \
      >"$copy/on-a/waves/3/prefill-2.timing.tsv"
    seal_process "$copy" on-a
}

mutate_startup_policy() {
    local copy=$1
    perl -i -pe '
      if (/DeepSeek-V4 full-context session worker started/) {
        s/mixed_cohort=true/mixed_cohort=false/; $seen++;
      }
      END {die "startup event absent\n" unless $seen == 1}
    ' "$copy/on-b/server.stderr"
    seal_process "$copy" on-b
}

mutate_power() {
    local copy=$1
    awk -F '\t' 'BEGIN{OFS="\t"} NR==2 {$2="battery"} {print}' \
      "$copy/off-a/power-measurement.tsv" >"$copy/off-a/power-measurement.tsv.tmp"
    mv "$copy/off-a/power-measurement.tsv.tmp" "$copy/off-a/power-measurement.tsv"
    seal_process "$copy" off-a
}

mutate_rss() {
    local copy=$1
    printf '121634817\n' >>"$copy/on-a/rss-kib"
    jq '.sampled_peak_rss_bytes=124554052608' "$copy/on-a/summary.json" \
      >"$copy/on-a/summary.json.tmp"
    mv "$copy/on-a/summary.json.tmp" "$copy/on-a/summary.json"
    seal_process "$copy" on-a
}

mutate_cohort_shape() {
    local copy=$1
    perl -i -pe '
      if (/DeepSeek-V4 cooperative prefill complete/ && !$seen) {
        s/rows_per_lane=128/rows_per_lane=127/; $seen=1;
      }
      END {die "cooperative event absent\n" unless $seen}
    ' "$copy/on-a/waves/2/server.delta.log"
    seal_process "$copy" on-a
}

expect_reject verdict-skip mutate_top '.verdict="skip"'
expect_reject threshold-widen mutate_top '.thresholds.semantic_sse_gap_ms=999999'
expect_reject false-speedup mutate_top '.result.wave_speedup=99'
expect_reject false-semantic-hash mutate_top '.equality.semantic_and_token_sha256=("f"*64)'
expect_reject false-summary-hash mutate_top \
  '.evidence.processes["on-a"].summary_sha256=("0"*64)'
expect_reject rebound-invalid-timing mutate_timing
expect_reject rebound-startup-policy mutate_startup_policy
expect_reject rebound-power-loss mutate_power
expect_reject rebound-rss-over-ceiling mutate_rss
expect_reject rebound-cohort-shape mutate_cohort_shape

echo "DeepSeek-V4 Mixed policy receipt mutations: 10/10 REJECTED"
