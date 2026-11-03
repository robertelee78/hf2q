#!/usr/bin/env bash
# Fail when a Mach-O LC_SYMTAB string pool (stroff) is not 8-byte aligned.
# Shared by the PR release build (ci.yml) and the exact standalone candidate
# build (standalone-candidate.yml) so a misaligned LINKEDIT is caught on the
# PR instead of after a release dispatch.
#
# usage: assert_macho_string_pool_aligned.sh BINARY DYLIB_ROOT
#   BINARY      the linked hf2q executable
#   DYLIB_ROOT  directory searched recursively for *.dylib; at least one
#               dylib must exist so a moved build root cannot pass vacuously
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BINARY DYLIB_ROOT" >&2
  exit 2
fi
binary=$1
dylib_root=$2

assert_macho_string_pool_aligned() {
  local macho=$1
  local stroff
  stroff=$(/usr/bin/otool -l "$macho" | awk '
    $1 == "cmd" && $2 == "LC_SYMTAB" { in_symtab = 1; next }
    in_symtab && $1 == "cmd" { in_symtab = 0 }
    in_symtab && $1 == "stroff" { print $2; exit }
  ')
  [[ -z "$stroff" || "$stroff" =~ ^[0-9]+$ ]]
  if [[ -n "$stroff" && $((stroff % 8)) -ne 0 ]]; then
    echo "misaligned Mach-O LINKEDIT string pool: $macho (stroff=$stroff)" >&2
    exit 1
  fi
}

[[ -d "$dylib_root" ]] || {
  echo "dylib root is not a directory: $dylib_root" >&2
  exit 1
}
dylib_count=0
while IFS= read -r -d '' dylib; do
  assert_macho_string_pool_aligned "$dylib"
  ((dylib_count += 1))
done < <(find "$dylib_root" -type f -name '*.dylib' -print0)
test "$dylib_count" -gt 0
test -x "$binary"
assert_macho_string_pool_aligned "$binary"
echo "Mach-O string pool alignment: PASS ($dylib_count dylibs + $(basename "$binary"))"
