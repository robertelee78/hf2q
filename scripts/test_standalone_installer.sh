#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd -P)
HF2Q_BIN=${HF2Q_BIN:-"$ROOT_DIR/target/debug/hf2q"}
TEMPLATE="$ROOT_DIR/scripts/install.sh.in"
RENDER="$ROOT_DIR/scripts/render_standalone_installer.sh"
RECORD_RENDER="$ROOT_DIR/scripts/render_standalone_release_record.sh"

[[ -x "$HF2Q_BIN" ]] || {
  echo "build hf2q first or set HF2Q_BIN" >&2
  exit 2
}

workspace=$(cd "$(mktemp -d "${TMPDIR:-/tmp}/hf2q-installer-test.XXXXXX")" && pwd -P)
trap 'rm -rf "$workspace"' EXIT
fixture="$workspace/release"
home="$workspace/home"
install_dir="$home/.local/bin"
state_root="$home/.hf2q"
asset="$fixture/hf2q-aarch64-apple-darwin"
installer="$workspace/install.sh"
release_record="$workspace/stable-aarch64-apple-darwin.json"
mkdir -p "$fixture"
cp "$HF2Q_BIN" "$asset"
chmod 0555 "$asset"
size=$(stat -f '%z' "$asset")
sha256=$(shasum -a 256 "$asset" | awk '{print $1}')
version=$($HF2Q_BIN --version | awk '{print $2}')

"$RECORD_RENDER" "$release_record" "$version" "$size" "$sha256" >/dev/null
expected_record=$(printf '{"kind":"hf2q.standalone-release","schema_version":1,"package":"hf2q","channel":"stable","target":"aarch64-apple-darwin","version":"%s","size":%s,"sha256":"%s"}' "$version" "$size" "$sha256")
[[ $(cat "$release_record") == "$expected_record" ]]
[[ $(stat -f '%Lp' "$release_record") == 444 ]]
# shellcheck disable=SC2016
grep -Fq '[ "$(/usr/bin/lipo -archs "$candidate" 2>/dev/null)" = arm64 ]' \
  "$TEMPLATE"
grep -Fq "[ \"\$macos_major\" -ge 14 ]" "$TEMPLATE"

"$RENDER" "$TEMPLATE" "$installer" "$version" "$size" "$sha256" ABCDE12345 us.hf2q.cli >/dev/null

trust_home="$workspace/trust-home"
trust_install="$trust_home/.local/bin"
if curl -fsSL "file://$installer" | \
  HOME="$trust_home" \
  HF2Q_INSTALL_DIR="$trust_install" \
  HF2Q_RELEASE_BASE_URL="file://$fixture" \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  sh >/dev/null 2>&1; then
  echo "unsigned local fixture unexpectedly passed Apple release trust" >&2
  exit 1
fi
[[ ! -e "$trust_install/hf2q" ]]
[[ ! -e "$trust_install/.hf2q-standalone.json" ]]

curl -fsSL "file://$installer" | \
  HOME="$home" \
  HF2Q_INSTALL_DIR="$install_dir" \
  HF2Q_RELEASE_BASE_URL="file://$fixture" \
  HF2Q_INSTALL_TEST_MODE=1 \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  sh

[[ -x "$install_dir/hf2q" ]]
[[ $(shasum -a 256 "$install_dir/hf2q" | awk '{print $1}') == "$sha256" ]]
[[ $(HOME="$home" "$install_dir/hf2q" --version) == "hf2q $version" ]]
[[ -f "$install_dir/.hf2q-standalone.json" ]]
[[ -f "$install_dir/.hf2q-standalone.lock" ]]
HOME="$home" "$install_dir/hf2q" --state-root "$state_root" \
  setup --accept-defaults >/dev/null
[[ -s "$state_root/config.toml" ]]
[[ $(stat -f '%Lp' "$state_root/config.toml") == 600 ]]
cmp -s "$state_root/config.toml" "$ROOT_DIR/src/setup/testdata/config_v2.toml"
mkdir -p "$state_root/models"
printf 'converted model\n' >"$state_root/models/model.gguf"
config_sha=$(shasum -a 256 "$state_root/config.toml" | awk '{print $1}')
model_sha=$(shasum -a 256 "$state_root/models/model.gguf" | awk '{print $1}')
[[ -f "$state_root/config.toml" ]]
[[ -f "$state_root/models/model.gguf" ]]

chmod 0755 "$asset"
printf 'corruption\n' >>"$asset"
if curl -fsSL "file://$installer" | \
  HOME="$home" \
  HF2Q_INSTALL_DIR="$install_dir" \
  HF2Q_RELEASE_BASE_URL="file://$fixture" \
  HF2Q_INSTALL_TEST_MODE=1 \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  sh >/dev/null 2>&1; then
  echo "corrupt immutable asset unexpectedly replaced the installation" >&2
  exit 1
fi
[[ $(shasum -a 256 "$install_dir/hf2q" | awk '{print $1}') == "$sha256" ]]

if HOME="$home" "$install_dir/hf2q" uninstall >/dev/null 2>&1; then
  echo "uninstall without confirmation unexpectedly succeeded" >&2
  exit 1
fi
[[ $(shasum -a 256 "$install_dir/hf2q" | awk '{print $1}') == "$sha256" ]]
HOME="$home" "$install_dir/hf2q" uninstall --yes >/dev/null
[[ ! -e "$install_dir/hf2q" ]]
[[ ! -e "$install_dir/.hf2q-standalone.json" ]]
[[ ! -e "$install_dir/.hf2q-standalone.lock" ]]
[[ -f "$state_root/config.toml" ]]
[[ -f "$state_root/models/model.gguf" ]]
[[ $(shasum -a 256 "$state_root/config.toml" | awk '{print $1}') == "$config_sha" ]]
[[ $(shasum -a 256 "$state_root/models/model.gguf" | awk '{print $1}') == "$model_sha" ]]

printf 'standalone installer fixture passed: %s (%s bytes)\n' "$sha256" "$size"
