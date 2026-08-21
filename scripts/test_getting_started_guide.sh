#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guide="$repo_root/docs/getting-started.md"
extra_guide="$repo_root/docs/hf2q+qwen3.8+ak+search-fetch-setup.md"
readme="$repo_root/README.md"
installer="$repo_root/scripts/install_opencode_web_stack.sh"
assets="$repo_root/scripts/opencode-web-stack"

fail() {
    echo "getting-started contract failed: $*" >&2
    exit 1
}

require_literal() {
    local file="$1"
    local literal="$2"
    grep -Fq -- "$literal" "$file" || fail "missing '$literal' in ${file#"$repo_root/"}"
}

reject_literal() {
    local file="$1"
    local literal="$2"
    if grep -Fq -- "$literal" "$file"; then
        fail "forbidden '$literal' remains in ${file#"$repo_root/"}"
    fi
}

# There is exactly one guide. Do not reintroduce a core/complete split.
[[ ! -e "$extra_guide" ]] || fail "second onboarding guide must not exist"

# Bind the complete issue-146 journey to the exact tested artifact and harness.
require_literal "$guide" "40d771ee15d826017f297261f5bedcf2c32cf4c2"
require_literal "$guide" "qwen38-abliterated-sft-hf2q-q4_k_m.gguf"
require_literal "$guide" "qwen38-abliterated-sft-hf2q-q4_k_m-mmproj.gguf"
require_literal "$guide" "1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a"
require_literal "$guide" "463b264713f8e081f0fae753c80d8089308e01b1e2ac0948dd9966d0711d8f1b"
require_literal "$guide" "--mmproj"
require_literal "$guide" "--port 8081"
require_literal "$guide" "lsof -nP -iTCP:8081 -sTCP:LISTEN"
require_literal "$guide" "kill -0 \"\$SERVER_PID\""
require_literal "$guide" "/v1/chat/completions"
require_literal "$guide" "data: \\[DONE\\]"
require_literal "$guide" "image_url"
require_literal "$guide" "ak setup --yes"
require_literal "$guide" "ak setup --opencode --yes"
require_literal "$guide" "--agent build"
require_literal "$guide" '"attachment": true'
require_literal "$guide" '"modalities": {"input": ["text", "image"], "output": ["text"]}'
require_literal "$guide" "preserving every existing agent, tool, permission,"
require_literal "$guide" "agent retains Bash, read/write/edit, task, skill, and"
require_literal "$guide" "perform a harmless proof: list the current"
require_literal "$guide" "HF2Q_DEFAULT_REPETITION_PENALTY=1.05"
require_literal "$guide" "HF2Q_DEFAULT_THINKING_TOKEN_BUDGET=2048"
require_literal "$guide" "HF2Q_QWEN_SPECULATION=auto"
require_literal "$guide" "install_opencode_web_stack.sh"
require_literal "$guide" "web_search"
require_literal "$guide" "web_fetch"
require_literal "$guide" "web_crawl"
require_literal "$guide" "web_extract"
require_literal "$guide" "bash -s -- --disable"
require_literal "$guide" "bash -s -- --uninstall"
require_literal "$guide" "SERVER_COMMAND=\"\$(ps -p \"\$SERVER_PID\" -o command="
require_literal "$guide" "\"--model \$MODEL\"*\"--port 8081\""

reject_literal "$guide" "ak setup --minimal --opencode"
reject_literal "$guide" '| .agent ='
reject_literal "$guide" '| .default_agent ='
reject_literal "$guide" '"tools": { "*": false }'
reject_literal "$guide" '"permission": "deny"'
reject_literal "$guide" 'pkill -f'
reject_literal "$guide" "hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT"
reject_literal "$guide" "Optional: convert the pair yourself"
reject_literal "$guide" "serves the text model only"
reject_literal "$readme" "downloads the model author's pinned Q5_K_M GGUF"
reject_literal "$readme" "The first serving path remains text-only"
reject_literal "$readme" "hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT"
reject_literal "$readme" "source-first core CLI journey"
reject_literal "$readme" "hf2q+qwen3.8+ak+search-fetch-setup.md"
require_literal "$readme" "](docs/getting-started.md)"

for asset in \
    web-search-fetch.js \
    server.py \
    stealth_fetch.py \
    requirements.txt \
    test_server.py \
    searxng-settings.yml
do
    [[ -s "$assets/$asset" ]] || fail "missing web-stack asset: $asset"
done
require_literal "$assets/searxng-settings.yml" '__SEARXNG_SECRET__'
reject_literal "$assets/searxng-settings.yml" 'bce60a2c2f73acf96d64eac04a6591d04'
require_literal "$assets/web-search-fetch.js" 'webfetch: false'
require_literal "$assets/web-search-fetch.js" 'webfetch: "deny"'
require_literal "$assets/web-search-fetch.js" 'normalizeJsonCssSchema'
reject_literal "$assets/web-search-fetch.js" '"*": false'
require_literal "$installer" '--no-build-isolation'
require_literal "$installer" 'BROWSER_SETUP_MARKER'
require_literal "$installer" 'LEGACY_BACKUPS'
require_literal "$installer" "BACKUP_DIR=\"\$FETCH_DIR/backups/\$BACKUP_TAG\""

bash -n "$installer"
bash "$repo_root/scripts/test_opencode_web_stack_lifecycle.sh"
bash "$repo_root/scripts/test_web_search_fetch_plugin.sh"
node --check "$assets/web-search-fetch.js"

guide_shell="$(mktemp -t hf2q-guide-shell.XXXXXX)"
pycache_dir="$(mktemp -d -t hf2q-guide-pycache.XXXXXX)"
cleanup() {
    rm -f "$guide_shell"
    rm -r "$pycache_dir"
}
trap cleanup EXIT

PYTHONPYCACHEPREFIX="$pycache_dir" python3 -m py_compile \
    "$assets/server.py" "$assets/stealth_fetch.py" "$assets/test_server.py"

awk '
    /^```bash$/ { in_shell = 1; next }
    /^```$/ && in_shell { in_shell = 0; print ""; next }
    in_shell { print }
' "$guide" > "$guide_shell"
bash -n "$guide_shell"

echo "single complete getting-started guide contract passed"
