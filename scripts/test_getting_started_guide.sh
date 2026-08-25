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

# --- Issue-146 prevention rules: bind the tested resolver journey and proof ---
# The public guide deliberately has no artifact-filename/download ritual. The
# exact repository + quant is the operator contract; hf2q resolves one immutable
# revision, verifies the selected payload, and prepares its matching projector.
model_operand="jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M"
require_literal "$guide" "hf2q serve $model_operand"
require_literal "$guide" "hf2q chat $model_operand"
require_literal "$guide" "hf2q serve list"
require_literal "$guide" "hf2q chat list"
require_literal "$guide" "resolves the repository to one immutable"
require_literal "$guide" "loads it automatically"
require_literal "$guide" 'v2-<hex(owner/repo)>/<commit>/'
# Generation is proven by a real client conversation, not by /readyz alone.
require_literal "$guide" "hf2q chat"
# Multimodal is served and proven with one simple image request.
require_literal "$guide" "image_url"
require_literal "$guide" "grep -i red"
# A failed pasteable proof must not exit the reader's interactive shell.
require_literal "$guide" $'(\nRED_PNG="'
require_literal "$guide" $'echo "vision check passed: $MODEL_ID saw red"\n)'
require_literal "$guide" "without closing your terminal"
# Full Agentic Kit: machine + project setup, OpenCode host wiring, then
# converge-and-verify. --minimal is never presented as equivalent.
require_literal "$guide" "ak setup"
require_literal "$guide" "ak setup --opencode"
require_literal "$guide" "ak sync"
# The OpenCode provider merge preserves existing configuration.
require_literal "$guide" "preserves every existing agent, tool, permission,"
require_literal "$guide" "opencode.json"
require_literal "$guide" "opencode --model"
# The local research stack and its four tool names.
require_literal "$guide" "install_opencode_web_stack.sh"
require_literal "$guide" "web_search"
require_literal "$guide" "web_fetch"
require_literal "$guide" "web_crawl"
require_literal "$guide" "web_extract"
require_literal "$guide" "--status"

# --- The guide must not reintroduce the slop it replaced ---
# No environment-variable ritual: qualified defaults ship in the product
# (setup-persisted profile + engine defaults), not in the guide.
for knob in \
    HF2Q_TQ_KV HF2Q_ENCODER_SESSION HF2Q_FFN_TERMINAL_K_BATCH \
    HF2Q_QWEN_SPECULATION HF2Q_DECODE_MVN HF2Q_DECODE_MV_EXT HF2Q_QWEN_GQA_Q2 \
    HF2Q_DEFAULT_REPETITION_PENALTY HF2Q_DEFAULT_THINKING_TOKEN_BUDGET \
    HF2Q_DEFAULT_TOOL_THINKING_TOKEN_BUDGET
do
    reject_literal "$guide" "$knob"
done
# No background-server apparatus: the guide serves in the foreground and
# stops with Ctrl-C, so PID files, nohup, and broad process kills are
# forbidden.
reject_literal "$guide" "nohup"
reject_literal "$guide" "SERVER_PID"
reject_literal "$guide" 'pkill -f'
# No harness destruction: never write agents, default agents, blanket tool
# removal, or blanket permission denial.
reject_literal "$guide" '| .agent ='
reject_literal "$guide" '| .default_agent ='
reject_literal "$guide" '"tools": { "*": false }'
reject_literal "$guide" '"permission": "deny"'
reject_literal "$guide" "ak setup --minimal --opencode"
# No source conversion is presented as the fast hosted-artifact journey.
reject_literal "$guide" "hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT"
reject_literal "$guide" "Optional: convert the pair yourself"
# The former manually pinned payload ritual must not creep back into the
# first-run path. Exact bytes remain receipt-bound by the resolver itself.
reject_literal "$guide" "40d771ee15d826017f297261f5bedcf2c32cf4c2"
reject_literal "$guide" "qwen38-abliterated-sft-hf2q-q4_k_m.gguf"
reject_literal "$guide" "shasum -a 256 -c hf2q-q4_k_m-SHA256SUMS.txt"

# --- README entry point stays pointed at the one guide ---
reject_literal "$readme" "downloads the model author's pinned Q5_K_M GGUF"
reject_literal "$readme" "hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT"
reject_literal "$readme" "hf2q+qwen3.8+ak+search-fetch-setup.md"
require_literal "$readme" "](docs/getting-started.md)"

# --- Web-stack assets the guide's installer step depends on ---
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

echo "getting-started guide contract passed"
