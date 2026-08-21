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
grep -Fq '/usr/sbin/sysctl -n hw.optional.arm64' "$TEMPLATE"
grep -Fq "[ \"\$macos_major\" -ge 14 ]" "$TEMPLATE"
# shellcheck disable=SC2016
grep -Fq '# The final `fi` makes this whole script one parse-before-execute command.' \
  "$TEMPLATE"
grep -Fq 'main "$@" </dev/null' "$TEMPLATE"
grep -Fq -- '--connect-timeout 15' "$TEMPLATE"
grep -Fq -- '--max-time 300' "$TEMPLATE"
grep -Fq -- '--max-redirs 3' "$TEMPLATE"
# shellcheck disable=SC2016
grep -Fq -- '--max-filesize "$release_size"' "$TEMPLATE"
# shellcheck disable=SC2016
grep -Fq 'ulimit -f "$file_limit_blocks"' "$TEMPLATE"
grep -Fq "Authority=Developer ID Certification Authority" "$TEMPLATE"
grep -Fq "Authority=Apple Root CA" "$TEMPLATE"
grep -Fq "flags=0x[0-9a-f]+\\(runtime\\)" "$TEMPLATE"
grep -Fq "grep -Eq '^Timestamp=.+$'" "$TEMPLATE"
grep -Fq -- '--check-notarization' "$TEMPLATE"
grep -Fq -- "--test-requirement '=notarized'" "$TEMPLATE"
if grep -Fq '/usr/sbin/spctl' "$TEMPLATE"; then
  echo "raw standalone installer must not apply the app-bundle-only spctl assessment" >&2
  exit 1
fi
if grep -Fq -- "--proto '=https,file'" "$TEMPLATE"; then
  echo "production installer must not allow file:// downloads" >&2
  exit 1
fi

"$RENDER" "$TEMPLATE" "$installer" "$version" "$size" "$sha256" ABCDE12345 us.hf2q.cli >/dev/null
bash -n "$installer"
/bin/sh -n "$installer"

trunc_home="$workspace/trunc-home"
trunc_install="$trunc_home/.local/bin"
main_line=$(grep -n '^main "\$@" </dev/null$' "$installer" | cut -d: -f1)
installer_bytes=$(stat -f '%z' "$installer")
sed -n "1,$((main_line - 1))p" "$installer" >"$workspace/truncated-before-main.sh"
sed -n "1,${main_line}p" "$installer" >"$workspace/truncated-after-main.sh"
head -c "$((installer_bytes - 2))" "$installer" >"$workspace/truncated-mid-fi.sh"
for truncated in \
  "$workspace/truncated-before-main.sh" \
  "$workspace/truncated-after-main.sh" \
  "$workspace/truncated-mid-fi.sh"; do
  if HOME="$trunc_home" \
    HF2Q_INSTALL_DIR="$trunc_install" \
    PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    /bin/sh "$truncated" >/dev/null 2>&1; then
    echo "truncated installer unexpectedly parsed: $truncated" >&2
    exit 1
  fi
  [[ ! -e "$trunc_install" ]]
done

probe_fixture="$workspace/stdin-probe-release"
probe_asset="$probe_fixture/hf2q-aarch64-apple-darwin"
probe_installer="$workspace/stdin-probe-installer.sh"
probe_log="$workspace/stdin-probe.log"
probe_home="$workspace/stdin-probe-home"
mkdir -p "$probe_fixture"
cp "$ROOT_DIR/scripts/testdata/installer_stdin_probe.sh" "$probe_asset"
chmod 0555 "$probe_asset"
probe_size=$(stat -f '%z' "$probe_asset")
probe_sha256=$(shasum -a 256 "$probe_asset" | awk '{print $1}')
"$RENDER" "$TEMPLATE" "$probe_installer" "$version" "$probe_size" \
  "$probe_sha256" ABCDE12345 us.hf2q.cli >/dev/null
printf 'pipeline bytes must not reach candidate\n' | \
  HOME="$probe_home" \
  HF2Q_INSTALL_DIR="$probe_home/.local/bin" \
  HF2Q_RELEASE_BASE_URL="file://$probe_fixture" \
  HF2Q_INSTALL_TEST_MODE=1 \
  HF2Q_STDIN_PROBE_LOG="$probe_log" \
  HF2Q_STDIN_PROBE_VERSION="$version" \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  /bin/sh "$probe_installer" >/dev/null
grep -Fx 'eof:--version' "$probe_log"
grep -Fx 'eof:__standalone-install' "$probe_log"
if grep -Fq 'captured:' "$probe_log"; then
  echo 'candidate inherited installer stdin' >&2
  exit 1
fi

trust_home="$workspace/trust-home"
trust_install="$trust_home/.local/bin"
if curl -fsSL "file://$installer" | \
  HOME="$trust_home" \
  HF2Q_INSTALL_DIR="$trust_install" \
  HF2Q_RELEASE_BASE_URL="file://$fixture" \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  sh >/dev/null 2>&1; then
  echo "production installer unexpectedly accepted a local release override" >&2
  exit 1
fi
[[ ! -e "$trust_install/hf2q" ]]
[[ ! -e "$trust_install/.hf2q-standalone.json" ]]

unsafe_install="$workspace/unsafe-bin"
mkdir -p "$unsafe_install"
chmod 0775 "$unsafe_install"
if curl -fsSL "file://$installer" | \
  HOME="$home" \
  HF2Q_INSTALL_DIR="$unsafe_install" \
  HF2Q_RELEASE_BASE_URL="file://$fixture" \
  HF2Q_INSTALL_TEST_MODE=1 \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  sh >/dev/null 2>&1; then
  echo "group-writable install directory unexpectedly accepted" >&2
  exit 1
fi
[[ -z $(find "$unsafe_install" -mindepth 1 -print -quit) ]]

curl -fsSL "file://$installer" | \
  env -u HOME \
  HF2Q_INSTALL_DIR="$install_dir" \
  HF2Q_RELEASE_BASE_URL="file://$fixture" \
  HF2Q_INSTALL_TEST_MODE=1 \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  sh

[[ -x "$install_dir/hf2q" ]]
[[ $(stat -f '%Lp' "$install_dir") == 755 ]]
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
dd if=/dev/zero bs=1024 count=2 >>"$asset" 2>/dev/null
if curl -fsSL "file://$installer" | \
  HOME="$home" \
  HF2Q_INSTALL_DIR="$install_dir" \
  HF2Q_RELEASE_BASE_URL="file://$fixture" \
  HF2Q_INSTALL_TEST_MODE=1 \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  sh >/dev/null 2>&1; then
  echo "oversized immutable asset unexpectedly replaced the installation" >&2
  exit 1
fi
[[ $(shasum -a 256 "$install_dir/hf2q" | awk '{print $1}') == "$sha256" ]]
[[ -z $(find "$install_dir" -name '.hf2q-download.*' -print -quit) ]]

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
