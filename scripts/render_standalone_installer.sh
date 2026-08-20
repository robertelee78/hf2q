#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 TEMPLATE OUTPUT VERSION SIZE SHA256 TEAM_ID IDENTIFIER" >&2
  exit 2
fi

template=$1
output=$2
version=$3
size=$4
sha256=$5
team_id=$6
identifier=$7

[[ -f "$template" ]] || {
  echo "installer template is missing: $template" >&2
  exit 2
}
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
[[ "$team_id" =~ ^[A-Z0-9]{10}$ ]] || {
  echo "Developer ID team must be 10 uppercase letters or digits" >&2
  exit 2
}
[[ "$identifier" =~ ^[A-Za-z0-9.-]+$ ]] || {
  echo "signing identifier contains unsupported characters" >&2
  exit 2
}
[[ ! -e "$output" && ! -L "$output" ]] || {
  echo "installer output already exists: $output" >&2
  exit 2
}

temporary="${output}.partial.$$"
trap 'rm -f "$temporary"' EXIT
sed \
  -e "s/@HF2Q_VERSION@/$version/g" \
  -e "s/@HF2Q_SIZE@/$size/g" \
  -e "s/@HF2Q_SHA256@/$sha256/g" \
  -e "s/@HF2Q_TEAM_ID@/$team_id/g" \
  -e "s/@HF2Q_IDENTIFIER@/$identifier/g" \
  "$template" >"$temporary"

if grep -q '@HF2Q_' "$temporary"; then
  echo "rendered installer retains an unresolved placeholder" >&2
  exit 1
fi
chmod 0555 "$temporary"
mv "$temporary" "$output"
trap - EXIT
printf '%s\n' "$output"
