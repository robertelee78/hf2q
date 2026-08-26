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

expect_accept() {
    local name=$1 copy
    copy=$(copy_evidence "$name")
    shift
    "$@" "$copy"
    if ! verify "$copy/receipt.json" >/dev/null 2>&1; then
        echo "receipt verifier rejected accept-preserving canary: $name" >&2
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

refresh_semantic_sha() {
    local copy=$1 semantic_sha
    semantic_sha=$(python3 - "$copy" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); values=[]
for replica in ("a","b"):
  for lane in range(1,5):
    value=json.loads((root/f"on-{replica}/decoder-prime/decoder-{lane}.canonical.json").read_text())
    values.append({k:value[k] for k in ("role_events","content","reasoning_content","tool_calls","finish_reason","usage","done_count")})
  for trial in range(1,6):
    for kind in ("decoder","prefill"):
      for lane in range(1,5):
        value=json.loads((root/f"on-{replica}/waves/{trial}/{kind}-{lane}.canonical.json").read_text())
        values.append({k:value[k] for k in ("role_events","content","reasoning_content","tool_calls","finish_reason","usage","done_count")})
payload=(json.dumps(values,sort_keys=True,separators=(",",":"))+"\n").encode()
print(hashlib.sha256(payload).hexdigest())
PY
)
    jq --arg sha "$semantic_sha" '.equality.semantic_and_token_sha256=$sha' \
      "$copy/receipt.json" >"$copy/receipt.json.tmp"
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
      if (/DeepSeek-V4 cooperative prefill complete/ && /bounded_mixed=true/ && !$seen) {
        s/rows_per_lane=128/rows_per_lane=127/; $seen=1;
      }
      END {die "cooperative event absent\n" unless $seen}
    ' "$copy/on-a/waves/2/server.delta.log"
    seal_process "$copy" on-a
}

mutate_mixed_discriminator() {
    local copy=$1
    perl -i -pe '
      if (/DeepSeek-V4 cooperative prefill complete/ && /bounded_mixed=true/ && !$seen) {
        s/bounded_mixed=true/bounded_mixed=false/;
        s/rows_per_lane_cap=128/rows_per_lane_cap=0/;
        $seen=1;
      }
      END {die "bounded Mixed event absent\n" unless $seen}
    ' "$copy/on-a/waves/2/server.delta.log"
    seal_process "$copy" on-a
}

mutate_decoder_cache() {
    local copy=$1 request_id
    request_id=$(jq -er '.decoder_request_ids[0]' "$copy/on-a/waves/3/wave.json")
    REQUEST_ID="$request_id" perl -i -pe '
      if (/DeepSeek-V4 prefill planned/ && /request_id=$ENV{REQUEST_ID}(?: |$)/ && !$seen) {
        s/cache=(?:live|grow-live|recovery-anchor|grow-recovery-anchor)/cache=fabricated/;
        $seen=1;
      }
      END {die "decoder cache plan absent\n" unless $seen}
    ' "$copy/on-a/waves/3/server.delta.log"
    seal_process "$copy" on-a
}

mutate_decoder_cache_accounting() {
    local copy=$1 request_id
    request_id=$(jq -er '.decoder_request_ids[0]' "$copy/on-a/waves/3/wave.json")
    REQUEST_ID="$request_id" perl -i -pe '
      if (/DeepSeek-V4 prefill planned/ && /request_id=$ENV{REQUEST_ID}(?: |$)/ && !$seen) {
        s/work_tokens=[0-9]+/work_tokens=999999999/; $seen=1;
      }
      END {die "decoder cache plan absent\n" unless $seen}
    ' "$copy/on-a/waves/3/server.delta.log"
    seal_process "$copy" on-a
}

mutate_decoder_client_cache() {
    local copy=$1 label canonical timed
    for label in off-a on-a; do
        canonical="$copy/$label/waves/3/decoder-1.canonical.json"
        timed="$copy/$label/waves/3/decoder-1.timed-sse"
        jq '.usage.prompt_tokens_details.cached_tokens=0' "$canonical" >"$canonical.tmp"
        mv "$canonical.tmp" "$canonical"
        perl -i -0pe '
          s/("prompt_tokens_details"\s*:\s*\{[^}]*"cached_tokens"\s*:\s*)[0-9]+/${1}0/
            or die "client cached-token usage absent\n";
        ' "$timed"
        seal_process "$copy" "$label"
    done
    refresh_semantic_sha "$copy"
}

mutate_prefill_cache() {
    local copy=$1 request_id
    request_id=$(jq -er '.prefill_request_ids[0]' "$copy/on-a/waves/3/wave.json")
    REQUEST_ID="$request_id" perl -i -pe '
      if (/DeepSeek-V4 prefill planned/ && /request_id=$ENV{REQUEST_ID}(?: |$)/ && !$seen) {
        s/cached_tokens=0/cached_tokens=1/;
        s/cache=(?:reset|grow-reset)/cache=live/;
        $seen=1;
      }
      END {die "cold prefill cache plan absent\n" unless $seen}
    ' "$copy/on-a/waves/3/server.delta.log"
    seal_process "$copy" on-a
}

mutate_prefill_admission() {
    local copy=$1 prefill_ids decoder_ids
    prefill_ids=$(jq -er '.prefill_request_ids|join(",")' "$copy/on-a/waves/3/wave.json")
    decoder_ids=$(jq -er '.decoder_request_ids|join(",")' "$copy/on-a/waves/3/wave.json")
    PREFILL_IDS="$prefill_ids" DECODER_IDS="$decoder_ids" perl -i -ne '
      BEGIN {
        %prefill=map {$_=>1} split /,/, $ENV{PREFILL_IDS};
        %decoder=map {$_=>1} split /,/, $ENV{DECODER_IDS};
      }
      if (/DeepSeek-V4 request started/ && /request_id=([0-9]+)/ && $prefill{$1}) {
        push @saved, $_; next;
      }
      print;
      if (!$inserted && /DeepSeek-V4 request complete/ && /request_id=([0-9]+)/ && $decoder{$1}) {
        die "four prefill admissions absent\n" unless @saved == 4;
        print @saved; $inserted=1;
      }
      END {die "decoder completion absent\n" unless $inserted}
    ' "$copy/on-a/waves/3/server.delta.log"
    seal_process "$copy" on-a
}

add_legitimate_pure_cohort() {
    local copy=$1
    printf '%s\n' \
      'INFO DeepSeek-V4 cooperative prefill complete bounded_mixed=false rows_per_lane_cap=0 lanes=4 rows_per_lane=512 aggregate_rows=2048' \
      >>"$copy/on-a/waves/3/server.delta.log"
    seal_process "$copy" on-a
}

mutate_prime_prompt() {
    local copy=$1
    jq '.messages[0].content += " drift"' "$copy/off-a/decoder-prime/decoder-1.request.json" \
      >"$copy/off-a/decoder-prime/decoder-1.request.json.tmp"
    mv "$copy/off-a/decoder-prime/decoder-1.request.json.tmp" \
      "$copy/off-a/decoder-prime/decoder-1.request.json"
    seal_process "$copy" off-a
}

mutate_contention_raw_owner() {
    local copy=$1
    local path="$copy/on-a/contention-measurement.tsv"
    awk -F '\t' 'BEGIN { OFS="\t" } NR == 2 { $4 = $4 + 1 } { print }' \
      "$path" >"$path.tmp"
    mv "$path.tmp" "$path"
    seal_process "$copy" on-a
}

expect_reject verdict-skip mutate_top '.verdict="skip"'
expect_reject threshold-widen mutate_top '.thresholds.semantic_sse_gap_ms=999999'
expect_reject stale-contention-policy mutate_top \
  '.environment.host_contention.policy="process-group-v1"'
expect_reject stale-contention-threshold mutate_top \
  '.environment.host_contention.maximum_foreign_cpu_percent=101'
expect_reject missing-contention-owner mutate_top \
  '.environment.host_contention.owner_pgid=0'
expect_reject noncontinuous-contention mutate_top \
  '.environment.host_contention.continuous=false'
expect_reject rebound-contention-row-owner mutate_contention_raw_owner
expect_reject false-speedup mutate_top '.result.wave_speedup=99'
expect_reject false-semantic-hash mutate_top '.equality.semantic_and_token_sha256=("f"*64)'
expect_reject false-summary-hash mutate_top \
  '.evidence.processes["on-a"].summary_sha256=("0"*64)'
expect_reject rebound-invalid-timing mutate_timing
expect_reject rebound-startup-policy mutate_startup_policy
expect_reject rebound-power-loss mutate_power
expect_reject rebound-rss-over-ceiling mutate_rss
expect_reject rebound-cohort-shape mutate_cohort_shape
expect_reject rebound-mixed-discriminator mutate_mixed_discriminator
expect_reject rebound-decoder-cache-action mutate_decoder_cache
expect_reject rebound-decoder-cache-accounting mutate_decoder_cache_accounting
expect_reject rebound-decoder-client-cache-loss mutate_decoder_client_cache
expect_reject rebound-prefill-cache-reuse mutate_prefill_cache
expect_reject rebound-prefill-admission-loss mutate_prefill_admission
expect_reject rebound-prime-prompt-drift mutate_prime_prompt
expect_accept rebound-legitimate-pure-cohort add_legitimate_pure_cohort

echo "DeepSeek-V4 Mixed policy receipt mutations: 23/23 (22 REJECTED, 1 ACCEPTED)"
