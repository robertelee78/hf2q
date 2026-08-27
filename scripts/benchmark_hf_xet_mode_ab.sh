#!/usr/bin/env bash
# Interleaved, cold-cache A/B proof for Xet adaptive defaults versus Xet's
# high-performance resource preset. Every artifact and receipt is retained.

set -euo pipefail

usage() {
  echo "usage: $0 <hf2q-binary> <absolute-results-dir> [rounds-per-arm]" >&2
  exit 2
}

[[ $# -eq 2 || $# -eq 3 ]] || usage

binary=$1
results_root=$2
rounds=${3:-3}
[[ -x "$binary" ]] || {
  echo "hf2q binary is not executable: $binary" >&2
  exit 2
}
[[ "$results_root" = /* ]] || {
  echo "results directory must be absolute: $results_root" >&2
  exit 2
}
[[ "$rounds" =~ ^[1-9][0-9]*$ ]] || {
  echo "rounds-per-arm must be a positive integer" >&2
  exit 2
}
[[ ! -e "$results_root" ]] || {
  echo "refusing to overwrite existing evidence: $results_root" >&2
  exit 2
}

# Other Xet tuning would make the arms incomparable. The harness owns the two
# mode variables and both cache roots; require every other Xet knob to be absent.
ambient_xet=()
while IFS= read -r variable; do
  case "$variable" in
    HF_XET_HIGH_PERFORMANCE|HF_XET_HP|HF_XET_CACHE) ;;
    *) ambient_xet+=("$variable") ;;
  esac
done < <(compgen -A variable HF_XET_)
if [[ ${HF_HUB_DISABLE_XET+x} ]]; then
  ambient_xet+=(HF_HUB_DISABLE_XET)
fi
if (( ${#ambient_xet[@]} > 0 )); then
  printf 'refusing ambient Xet tuning; unset before benchmarking:' >&2
  printf ' %s' "${ambient_xet[@]}" >&2
  printf '\n' >&2
  exit 2
fi

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

# Two arms retain one complete payload per round. Leave 10% headroom plus
# 16 GiB for cache metadata, incomplete state, and HP-mode transient writes.
retained_runs=$((rounds * 2))
required_bytes=$((expected_bytes * retained_runs))
required_bytes=$((required_bytes + required_bytes / 10 + 16 * 1024 * 1024 * 1024))
available_kib=$(df -Pk "$results_parent" | awk 'NR == 2 {print $4}')
available_bytes=$((available_kib * 1024))
if (( available_bytes < required_bytes )); then
  echo "insufficient space for retained A/B evidence: need $required_bytes bytes, have $available_bytes" >&2
  exit 1
fi

mkdir "$results_root"
results_file="$results_root/results.tsv"
summary_file="$results_root/summary.tsv"
manifest_file="$results_root/manifest.txt"

binary_version=$("$binary" --version)
binary_sha256=$(shasum -a 256 "$binary" | awk '{print $1}')
{
  echo "schema=hf2q.hf-xet-download-mode-ab.v1"
  echo "started_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "host_arch=$(uname -m)"
  echo "os_version=$(sw_vers -productVersion)"
  echo "repository=$repository"
  echo "revision=$revision"
  echo "artifact=$artifact"
  echo "expected_bytes=$expected_bytes"
  echo "expected_sha256=$expected_sha256"
  echo "rounds_per_arm=$rounds"
  echo "binary_version=$binary_version"
  echo "binary_sha256=$binary_sha256"
  echo "arms=adaptive-default,high-performance"
  echo "order=interleaved-and-alternated"
} > "$manifest_file"
printf 'round\torder\tmode\telapsed_seconds\tmib_per_second\tuser_seconds\tsystem_seconds\tmax_rss_bytes\tbytes\tsha256\tcache_root\n' > "$results_file"

run_arm() {
  local round=$1
  local order=$2
  local mode=$3
  local run_root="$results_root/round-$round-$order-$mode"
  local stdout_file="$run_root/stdout.log"
  local stderr_file="$run_root/stderr.log"
  local timing_file="$run_root/timing.txt"
  local downloaded_path actual_bytes actual_sha256 elapsed_seconds
  local user_seconds system_seconds max_rss_bytes mib_per_second

  mkdir "$run_root" "$run_root/hub" "$run_root/xet"
  echo "starting round=$round order=$order mode=$mode at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  if [[ "$mode" = high-performance ]]; then
    /usr/bin/time -p -l -o "$timing_file" env \
      -u HF_HUB_DISABLE_XET \
      -u HF_XET_HP \
      HF_XET_HIGH_PERFORMANCE=1 \
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
  else
    /usr/bin/time -p -l -o "$timing_file" env \
      -u HF_HUB_DISABLE_XET \
      -u HF_XET_HP \
      HF_XET_HIGH_PERFORMANCE=0 \
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
  fi

  downloaded_path=$(awk 'NF {line=$0} END {print line}' "$stdout_file")
  [[ -f "$downloaded_path" ]] || {
    echo "download command did not return a regular file: $downloaded_path" >&2
    exit 1
  }
  actual_bytes=$(wc -c < "$downloaded_path" | tr -d '[:space:]')
  actual_sha256=$(shasum -a 256 "$downloaded_path" | awk '{print $1}')
  [[ "$actual_bytes" = "$expected_bytes" ]] || {
    echo "byte mismatch for round=$round mode=$mode: $actual_bytes" >&2
    exit 1
  }
  [[ "$actual_sha256" = "$expected_sha256" ]] || {
    echo "digest mismatch for round=$round mode=$mode: $actual_sha256" >&2
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$round" "$order" "$mode" "$elapsed_seconds" "$mib_per_second" \
    "$user_seconds" "$system_seconds" "$max_rss_bytes" "$actual_bytes" \
    "$actual_sha256" "$run_root" >> "$results_file"
  echo "completed round=$round mode=$mode elapsed=${elapsed_seconds}s throughput=${mib_per_second}MiB/s"
}

for ((round = 1; round <= rounds; round++)); do
  if (( round % 2 == 1 )); then
    run_arm "$round" 1 adaptive-default
    run_arm "$round" 2 high-performance
  else
    run_arm "$round" 1 high-performance
    run_arm "$round" 2 adaptive-default
  fi
done

printf 'mode\tmedian_elapsed_seconds\tmedian_mib_per_second\tmedian_max_rss_bytes\n' > "$summary_file"
for mode in adaptive-default high-performance; do
  elapsed_sorted="$results_root/$mode-elapsed-sorted.txt"
  rss_sorted="$results_root/$mode-rss-sorted.txt"
  awk -F '\t' -v mode="$mode" 'NR > 1 && $3 == mode {print $4}' "$results_file" | sort -n > "$elapsed_sorted"
  awk -F '\t' -v mode="$mode" 'NR > 1 && $3 == mode {print $8}' "$results_file" | sort -n > "$rss_sorted"
  if (( rounds % 2 == 1 )); then
    median_elapsed=$(sed -n "$((rounds / 2 + 1))p" "$elapsed_sorted")
    median_rss=$(sed -n "$((rounds / 2 + 1))p" "$rss_sorted")
  else
    median_elapsed=$(awk -v first="$((rounds / 2))" -v second="$((rounds / 2 + 1))" \
      'NR == first {a=$1} NR == second {printf "%.3f\n", (a + $1) / 2}' "$elapsed_sorted")
    median_rss=$(awk -v first="$((rounds / 2))" -v second="$((rounds / 2 + 1))" \
      'NR == first {a=$1} NR == second {printf "%.0f\n", (a + $1) / 2}' "$rss_sorted")
  fi
  median_mib_per_second=$(awk -v bytes="$expected_bytes" -v seconds="$median_elapsed" \
    'BEGIN {printf "%.3f", bytes / 1048576 / seconds}')
  printf '%s\t%s\t%s\t%s\n' \
    "$mode" "$median_elapsed" "$median_mib_per_second" "$median_rss" >> "$summary_file"
done

echo "completed_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$manifest_file"
echo "results=$results_file"
echo "summary=$summary_file"
cat "$summary_file"
