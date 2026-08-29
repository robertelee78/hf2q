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
    "$test_home/.config/opencode/commands" \
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

cat > "$fake_bin/curl" <<'EOF'
#!/usr/bin/env bash
case "$*" in
    *127.0.0.1:11235/healthz*)
        printf '%s\n' '{"ok":true,"browser_warm":true,"stealth_installed":true}'
        ;;
    *127.0.0.1:11235/fetch*)
        printf '%s\n' '{"ok":true,"markdown":"example"}'
        ;;
    *127.0.0.1:11235/search-fallback*)
        if [[ "${FAKE_FALLBACK_OK:-0}" -eq 1 ]]; then
            printf '%s\n' '{"ok":true,"provider":"bing-browser-fallback","via":"browser","results":[{"title":"About Unicornscan","url":"https://unicornscan.org/about"}]}'
        else
            printf '%s\n' '{"ok":false,"error":"forced failure","results":[]}'
        fi
        ;;
    *127.0.0.1:8888/search*)
        printf '%s\n' '{"results":[],"unresponsive_engines":[["bing","forced failure"]]}'
        ;;
    *)
        exit 22
        ;;
esac
EOF
chmod 0755 "$fake_bin/curl"

plugin="$test_home/.config/opencode/plugins/web-search-fetch.js"
command_file="$test_home/.config/opencode/commands/search.md"
searx_plist="$test_home/Library/LaunchAgents/com.opencode.searxng.plist"
fetch_plist="$test_home/Library/LaunchAgents/com.opencode.crawl4ai.plist"
printf 'plugin\n' > "$plugin"
printf 'command\n' > "$command_file"
printf 'plist\n' > "$searx_plist"
printf 'plist\n' > "$fetch_plist"
printf 'state\n' > "$test_home/.local/opt/searxng/marker"
printf 'state\n' > "$test_home/.local/opt/crawl4ai-server/marker"
printf 'log\n' > "$test_home/.local/state/searxng.log"
printf 'log\n' > "$test_home/.local/state/crawl4ai.log"
printf 'backup\n' > "$plugin.20260821120000.bak"
printf 'backup\n' > "$command_file.20260821120000.bak"
printf 'backup\n' > "$searx_plist.20260821120000.bak"
printf 'backup\n' > "$fetch_plist.20260821120000.bak"

if HOME="$test_home" PATH="$fake_bin:$PATH" "$installer" --status >"$test_root/status-failed" 2>&1; then
    echo "status accepted a broken primary and fallback" >&2
    exit 1
fi
grep -Fq "browser discovery fallback: FAILED" "$test_root/status-failed"
HOME="$test_home" PATH="$fake_bin:$PATH" FAKE_FALLBACK_OK=1 \
    "$installer" --status >"$test_root/status-fallback"
grep -Fq "browser discovery fallback: 1 results via browser" "$test_root/status-fallback"
grep -Fq "status: healthy" "$test_root/status-fallback"

HOME="$test_home" PATH="$fake_bin:$PATH" "$installer" --disable >/dev/null
[[ ! -e "$plugin" ]]
[[ -f "$plugin.disabled" ]]
[[ ! -e "$command_file" ]]
[[ -f "$command_file.disabled" ]]

HOME="$test_home" PATH="$fake_bin:$PATH" "$installer" --enable >/dev/null
[[ -f "$plugin" ]]
[[ ! -e "$plugin.disabled" ]]
[[ -f "$command_file" ]]
[[ ! -e "$command_file.disabled" ]]

HOME="$test_home" PATH="$fake_bin:$PATH" "$installer" --uninstall >/dev/null
[[ ! -e "$plugin" ]]
[[ ! -e "$command_file" ]]
[[ ! -e "$searx_plist" ]]
[[ ! -e "$fetch_plist" ]]
[[ ! -e "$test_home/.local/opt/searxng" ]]
[[ ! -e "$test_home/.local/opt/crawl4ai-server" ]]

trash_dir="$(find "$test_home/.Trash" -mindepth 1 -maxdepth 1 -type d -name 'hf2q-opencode-web-stack-*' -print -quit)"
[[ -n "$trash_dir" ]]
[[ -f "$trash_dir/web-search-fetch.js" ]]
[[ -f "$trash_dir/search.md" ]]
[[ -f "$trash_dir/com.opencode.searxng.plist" ]]
[[ -f "$trash_dir/com.opencode.crawl4ai.plist" ]]
[[ -f "$trash_dir/web-search-fetch.js.20260821120000.bak" ]]
[[ -f "$trash_dir/search.md.20260821120000.bak" ]]
[[ -f "$trash_dir/com.opencode.searxng.plist.20260821120000.bak" ]]
[[ -f "$trash_dir/com.opencode.crawl4ai.plist.20260821120000.bak" ]]
[[ -d "$trash_dir/searxng" ]]
[[ -d "$trash_dir/crawl4ai-server" ]]

grep -Fq "bootout gui/$(id -u)" "$test_home/launchctl.calls"
grep -Fq "bootstrap gui/$(id -u)" "$test_home/launchctl.calls"

echo "OpenCode web-stack lifecycle contract passed"
