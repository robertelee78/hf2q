#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <parity-summary.json> <parity-directory>" >&2
  exit 2
fi

summary=$1
parity_dir=$2
for command in awk cmp find jq mktemp shasum sort; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ -s "$summary" && -f "$summary" && ! -L "$summary" ]] || {
  echo "Gemma parity summary must be a nonempty regular file: $summary" >&2
  exit 1
}
[[ -d "$parity_dir" && ! -L "$parity_dir" ]] || {
  echo "Gemma parity directory must be a real directory: $parity_dir" >&2
  exit 1
}

tmp_root=$(mktemp -d)
cleanup_parity_verifier() {
  rm -rf "$tmp_root"
}
trap cleanup_parity_verifier EXIT
summary_one="$tmp_root/summary.json"
expected_names="$tmp_root/expected-names.txt"
actual_paths="$tmp_root/actual-paths.txt"
actual_names="$tmp_root/actual-names.txt"

jq -e -s '
  if length == 1 and (.[0] | type) == "object" and .[0].status == "pass" then
    .[0]
  else
    error("Gemma parity receipt must contain exactly one passing JSON object")
  end
' "$summary" >"$summary_one"

entries=(
  n4_log_sha256 n4.log
  n8_log_sha256 n8.log
  n8_seed_budget_log_sha256 n8-seed-budget.log
  n8_tiny_hybrid_log_sha256 n8-tiny-hybrid.log
  n8_tiny_full_tq_log_sha256 n8-tiny-full-tq.log
  boundary_tail_log_sha256 boundary-tail.log
  long_resume_log_sha256 long-resume.log
)
for ((i = 1; i < ${#entries[@]}; i += 2)); do
  printf '%s\n' "${entries[$i]}"
done | sort >"$expected_names"

if ! find "$parity_dir" -maxdepth 1 -name '*.log' -print >"$actual_paths"; then
  echo "failed to enumerate Gemma parity logs" >&2
  exit 1
fi
while IFS= read -r path; do
  printf '%s\n' "${path##*/}"
done <"$actual_paths" | sort >"$actual_names"
cmp -s "$expected_names" "$actual_names" || {
  echo "Gemma parity log inventory is not the exact seven-file contract" >&2
  exit 1
}

for ((i = 0; i < ${#entries[@]}; i += 2)); do
  field=${entries[$i]}
  name=${entries[$((i + 1))]}
  path="$parity_dir/$name"
  [[ -s "$path" && -f "$path" && ! -L "$path" ]] || {
    echo "Gemma parity log must be a nonempty regular file: $name" >&2
    exit 1
  }
  expected=$(jq -er --arg field "$field" \
    '.[$field] | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    "$summary_one")
  actual=$(shasum -a 256 "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "Gemma parity log digest mismatch: $name" >&2
    exit 1
  }
done

echo "Gemma parity receipt verified"
