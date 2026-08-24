#!/usr/bin/env bash
# Exact-binary, no-network smoke for the frictionless repository UX.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <hf2q-binary> <temporary-root-prefix>" >&2
  exit 2
fi

binary=$1
root_prefix=$2
test -x "$binary"
binary_dir=$(cd "$(dirname "$binary")" && pwd -P)
binary="$binary_dir/$(basename "$binary")"
test_root=$(mktemp -d "${root_prefix}.XXXXXX")
trap 'rm -rf -- "$test_root"' EXIT
mkdir -p "$test_root/home" "$test_root/data" "$test_root/cache" \
  "$test_root/state"
printf 'this = [is not toml\n' > "$test_root/state/config.toml"

run_isolated() {
  HOME="$test_root/home" \
  XDG_DATA_HOME="$test_root/data" \
  XDG_CACHE_HOME="$test_root/cache" \
  HF_HOME="$test_root/cache/huggingface" \
  HF_HUB_OFFLINE=1 \
    "$binary" "$@"
}

run_isolated --state-root "$test_root/state" serve list \
  > "$test_root/serve.list"
run_isolated --state-root "$test_root/state" chat list \
  > "$test_root/chat.list"
cmp "$test_root/serve.list" "$test_root/chat.list"
test ! -e "$test_root/data/hf2q/models"

for command in serve chat convert; do
  run_isolated "$command" owner/model:Q4_K_M --help >/dev/null
  run_isolated "$command" owner/model --help >/dev/null
done
test ! -e "$test_root/data/hf2q/models"
