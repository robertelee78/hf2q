#!/usr/bin/env bash
# Reproducible cold-cache proof of hf2q's native-Xet download path. Every run
# receives private Hub and Xet cache roots; no downloaded evidence is deleted.

set -euo pipefail

usage() {
  echo "usage: $0 <hf2q-binary> <absolute-results-dir> [runs]" >&2
  exit 2
}

[[ $# -eq 2 || $# -eq 3 ]] || usage

binary=$1
results_root=$2
runs=${3:-3}

[[ -x "$binary" ]] || {
  echo "hf2q binary is not executable: $binary" >&2
  exit 2
}
[[ "$results_root" = /* ]] || {
  echo "results directory must be absolute: $results_root" >&2
  exit 2
}
[[ "$runs" =~ ^[1-9][0-9]*$ ]] || {
  echo "runs must be a positive integer" >&2
  exit 2
}
[[ ! -e "$results_root" ]] || {
  echo "refusing to overwrite existing evidence: $results_root" >&2
  exit 2
}

repository='jenerallee78/Qwen3.8-27B-Abliterated-SFT'
revision='0a72776892f98db49381fdf69f4b9982222ec9dc'
artifact='gguf/qwen38-abliterated-sft-hf2q-q4_k_m.gguf'
expected_bytes=16810714944
expected_sha256='1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a'
quant='Q4_K_M'

results_parent=$(dirname "$results_root")
[[ -d "$results_parent" ]] || {
  echo "results parent does not exist: $results_parent" >&2
  exit 2
}

# Retain every completed artifact. Leave 10% headroom plus 8 GiB for cache
# metadata and incomplete state rather than beginning an unprovable run.
required_bytes=$((expected_bytes * runs))
required_bytes=$((required_bytes + required_bytes / 10 + 8 * 1024 * 1024 * 1024))
available_kib=$(df -Pk "$results_parent" | awk 'NR == 2 {print $4}')
available_bytes=$((available_kib * 1024))
if (( available_bytes < required_bytes )); then
  echo "insufficient space for retained benchmark evidence: need $required_bytes bytes, have $available_bytes" >&2
  exit 1
fi

mkdir "$results_root"
results_file="$results_root/results.tsv"
summary_file="$results_root/summary.tsv"
manifest_file="$results_root/manifest.txt"

binary_version=$("$binary" --version)
binary_sha256=$(shasum -a 256 "$binary" | awk '{print $1}')

{
  echo "schema=hf2q.hf-xet-download-benchmark.v1"
  echo "started_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "host_arch=$(uname -m)"
  echo "os_version=$(sw_vers -productVersion)"
  echo "repository=$repository"
  echo "revision=$revision"
  echo "artifact=$artifact"
  echo "expected_bytes=$expected_bytes"
  echo "expected_sha256=$expected_sha256"
  echo "runs=$runs"
  echo "binary_version=$binary_version"
  echo "binary_sha256=$binary_sha256"
  echo "xet_mode=adaptive-default"
} > "$manifest_file"

printf 'trial\telapsed_seconds\tmib_per_second\tuser_seconds\tsystem_seconds\tmax_rss_bytes\tbytes\tsha256\tcache_root\n' > "$results_file"

for ((trial = 1; trial <= runs; trial++)); do
  run_root="$results_root/$trial"
  stdout_file="$run_root/stdout.log"
  stderr_file="$run_root/stderr.log"
  timing_file="$run_root/timing.txt"
  mkdir "$run_root" "$run_root/hub" "$run_root/xet"
  echo "starting native-Xet trial=$trial at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  /usr/bin/time -p -l -o "$timing_file" env \
    -u HF_HUB_DISABLE_XET \
    -u HF_XET_HIGH_PERFORMANCE \
    -u HF_XET_HP \
    HF_HUB_CACHE="$run_root/hub" \
    HF_XET_CACHE="$run_root/xet" \
    "$binary" __fetch-hub-gguf \
      --repository "$repository" \
      --revision "$revision" \
      --artifact "$artifact" \
      --bytes "$expected_bytes" \
      --sha256 "$expected_sha256" \
      --quant "$quant" \
      > "$stdout_file" 2> "$stderr_file"

  downloaded_path=$(awk 'NF {line=$0} END {print line}' "$stdout_file")
  [[ -f "$downloaded_path" ]] || {
    echo "download command did not return a regular file: $downloaded_path" >&2
    exit 1
  }
  actual_bytes=$(wc -c < "$downloaded_path" | tr -d '[:space:]')
  actual_sha256=$(shasum -a 256 "$downloaded_path" | awk '{print $1}')
  [[ "$actual_bytes" = "$expected_bytes" ]] || {
    echo "byte mismatch for trial=$trial: $actual_bytes" >&2
    exit 1
  }
  [[ "$actual_sha256" = "$expected_sha256" ]] || {
    echo "digest mismatch for trial=$trial: $actual_sha256" >&2
    exit 1
  }
  elapsed_seconds=$(awk '$1 == "real" {print $2}' "$timing_file")
  user_seconds=$(awk '$1 == "user" {print $2}' "$timing_file")
  system_seconds=$(awk '$1 == "sys" {print $2}' "$timing_file")
  max_rss_bytes=$(awk '/maximum resident set size/ {print $1}' "$timing_file")
  [[ "$elapsed_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 1
  [[ "$user_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 1
  [[ "$system_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 1
  [[ "$max_rss_bytes" =~ ^[0-9]+$ ]] || exit 1
  mib_per_second=$(awk -v bytes="$actual_bytes" -v seconds="$elapsed_seconds" \
    'BEGIN {printf "%.3f", bytes / 1048576 / seconds}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$trial" "$elapsed_seconds" "$mib_per_second" "$user_seconds" \
    "$system_seconds" "$max_rss_bytes" "$actual_bytes" "$actual_sha256" \
    "$run_root" >> "$results_file"
  echo "completed trial=$trial elapsed=${elapsed_seconds}s throughput=${mib_per_second}MiB/s"
done

sorted="$results_root/elapsed-sorted.txt"
awk -F '\t' 'NR > 1 {print $2}' "$results_file" | sort -n > "$sorted"
if (( runs % 2 == 1 )); then
  median=$(sed -n "$((runs / 2 + 1))p" "$sorted")
else
  median=$(awk -v first="$((runs / 2))" -v second="$((runs / 2 + 1))" \
    'NR == first {a=$1} NR == second {printf "%.3f\n", (a + $1) / 2}' "$sorted")
fi
median_mib_per_second=$(awk -v bytes="$expected_bytes" -v seconds="$median" \
  'BEGIN {printf "%.3f", bytes / 1048576 / seconds}')
{
  printf 'median_elapsed_seconds\tmedian_mib_per_second\n'
  printf '%s\t%s\n' "$median" "$median_mib_per_second"
} > "$summary_file"

echo "completed_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$manifest_file"
echo "results=$results_file"
echo "summary=$summary_file"
cat "$summary_file"
