#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 OUTPUT VERSION SIZE SHA256" >&2
  exit 2
fi

output=$1
version=$2
size=$3
sha256=$4

[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || {
  echo "release version must be canonical stable SemVer" >&2
  exit 2
}
[[ "$size" =~ ^[1-9][0-9]*$ ]] || {
  echo "release size must be a positive decimal integer" >&2
  exit 2
}
[[ "$sha256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "release SHA-256 must be 64 lowercase hexadecimal characters" >&2
  exit 2
}
[[ ! -e "$output" && ! -L "$output" ]] || {
  echo "release-record output already exists: $output" >&2
  exit 2
}

temporary="${output}.partial.$$"
trap 'rm -f "$temporary"' EXIT
printf '{"kind":"hf2q.standalone-release","schema_version":1,"package":"hf2q","channel":"stable","target":"aarch64-apple-darwin","version":"%s","size":%s,"sha256":"%s"}\n' \
  "$version" "$size" "$sha256" >"$temporary"
chmod 0444 "$temporary"
mv "$temporary" "$output"
trap - EXIT
printf '%s\n' "$output"
