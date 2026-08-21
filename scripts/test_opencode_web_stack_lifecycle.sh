#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
installer="$repo_root/scripts/install_opencode_web_stack.sh"
test_root="$(mktemp -d -t hf2q-web-lifecycle.XXXXXX)"
test_home="$test_root/home"
fake_bin="$test_root/bin"
mkdir -p \
    "$fake_bin" \
    "$test_home/.config/opencode/plugins" \
    "$test_home/.local/opt/searxng" \
    "$test_home/.local/opt/crawl4ai-server" \
    "$test_home/.local/state" \
    "$test_home/Library/LaunchAgents"

cleanup() {
    rm -r "$test_root"
}
trap cleanup EXIT

cat > "$fake_bin/launchctl" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$HOME/launchctl.calls"
exit 0
EOF
chmod 0755 "$fake_bin/launchctl"

plugin="$test_home/.config/opencode/plugins/web-search-fetch.js"
searx_plist="$test_home/Library/LaunchAgents/com.opencode.searxng.plist"
fetch_plist="$test_home/Library/LaunchAgents/com.opencode.crawl4ai.plist"
printf 'plugin\n' > "$plugin"
printf 'plist\n' > "$searx_plist"
printf 'plist\n' > "$fetch_plist"
printf 'state\n' > "$test_home/.local/opt/searxng/marker"
printf 'state\n' > "$test_home/.local/opt/crawl4ai-server/marker"
printf 'log\n' > "$test_home/.local/state/searxng.log"
printf 'log\n' > "$test_home/.local/state/crawl4ai.log"
printf 'backup\n' > "$plugin.20260821120000.bak"
printf 'backup\n' > "$searx_plist.20260821120000.bak"
printf 'backup\n' > "$fetch_plist.20260821120000.bak"

HOME="$test_home" PATH="$fake_bin:$PATH" "$installer" --disable >/dev/null
[[ ! -e "$plugin" ]]
[[ -f "$plugin.disabled" ]]

HOME="$test_home" PATH="$fake_bin:$PATH" "$installer" --enable >/dev/null
[[ -f "$plugin" ]]
[[ ! -e "$plugin.disabled" ]]

HOME="$test_home" PATH="$fake_bin:$PATH" "$installer" --uninstall >/dev/null
[[ ! -e "$plugin" ]]
[[ ! -e "$searx_plist" ]]
[[ ! -e "$fetch_plist" ]]
[[ ! -e "$test_home/.local/opt/searxng" ]]
[[ ! -e "$test_home/.local/opt/crawl4ai-server" ]]

trash_dir="$(find "$test_home/.Trash" -mindepth 1 -maxdepth 1 -type d -name 'hf2q-opencode-web-stack-*' -print -quit)"
[[ -n "$trash_dir" ]]
[[ -f "$trash_dir/web-search-fetch.js" ]]
[[ -f "$trash_dir/com.opencode.searxng.plist" ]]
[[ -f "$trash_dir/com.opencode.crawl4ai.plist" ]]
[[ -f "$trash_dir/web-search-fetch.js.20260821120000.bak" ]]
[[ -f "$trash_dir/com.opencode.searxng.plist.20260821120000.bak" ]]
[[ -f "$trash_dir/com.opencode.crawl4ai.plist.20260821120000.bak" ]]
[[ -d "$trash_dir/searxng" ]]
[[ -d "$trash_dir/crawl4ai-server" ]]

grep -Fq "bootout gui/$(id -u)" "$test_home/launchctl.calls"
grep -Fq "bootstrap gui/$(id -u)" "$test_home/launchctl.calls"

echo "OpenCode web-stack lifecycle contract passed"
