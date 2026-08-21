#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guide="$repo_root/docs/getting-started.md"
readme="$repo_root/README.md"
architecture="$repo_root/docs/ARCHITECTURE.md"
setup_doc="$repo_root/docs/setup.md"

fail() {
    echo "getting-started guide check failed: $*" >&2
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
        fail "stale or out-of-scope '$literal' remains in ${file#"$repo_root/"}"
    fi
}

require_literal "$guide" "curl -fsSL https://hf2q.us/install.sh | sh"
require_literal "$guide" "hf2q --version"
require_literal "$guide" "validated against hf2q 0.1.8"
require_literal "$guide" "hf2q setup"
require_literal "$guide" "hf2q setup --accept-defaults"
require_literal "$guide" "hf2q doctor"
require_literal "$guide" "hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT"
require_literal "$guide" "--revision 08c2f075b43bc06456382db6b918a3dcabdcf4dd"
require_literal "$guide" "--quant q4_k_m"
require_literal "$guide" "hf2q serve --model \"\$MODEL\""
require_literal "$guide" "hf2q chat --url http://127.0.0.1:8081/v1"
require_literal "$guide" "http://127.0.0.1:8081/v1/models"
require_literal "$guide" "http://127.0.0.1:8081/v1/chat/completions"
require_literal "$guide" "server later with Ctrl-C"
require_literal "$guide" "hf2q update --check"
require_literal "$guide" "hf2q uninstall --yes"

reject_literal "$guide" "qwen38-abliterated-sft-q5_k_m.gguf"
reject_literal "$guide" "npm install -"
reject_literal "$guide" "brew install"
reject_literal "$guide" "ak setup"
reject_literal "$guide" "search-fetch-setup.md"
reject_literal "$guide" "nohup"
reject_literal "$guide" "pkill -f"
reject_literal "$guide" "docker run"

require_literal "$readme" "**[Get started with hf2q and Qwen3.8](docs/getting-started.md)**"
require_literal "$readme" "Those third-party integrations are outside ADR-045"
require_literal "$architecture" '| Follow the supported first-run journey | `docs/getting-started.md` |'
require_literal "$setup_doc" "[Get started with hf2q and Qwen3.8](getting-started.md)"

core_line="$(grep -nF '**[Get started with hf2q and Qwen3.8](docs/getting-started.md)**' "$readme" | head -n 1 | cut -d: -f1)"
optional_line="$(grep -nF '[separate optional integration guide](docs/hf2q+qwen3.8+ak+search-fetch-setup.md)' "$readme" | head -n 1 | cut -d: -f1)"
[[ -n "$core_line" && -n "$optional_line" && "$core_line" -lt "$optional_line" ]] ||
    fail "the native core journey must remain primary in README"

convert_line="$(grep -nF "hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT" "$guide" | head -n 1 | cut -d: -f1)"
opencode_line="$(grep -nF "Connect an existing OpenCode installation" "$guide" | head -n 1 | cut -d: -f1)"
[[ -n "$convert_line" && -n "$opencode_line" && "$convert_line" -lt "$opencode_line" ]] ||
    fail "native conversion must remain before the optional client section"

syntax_file="$(mktemp "${TMPDIR:-/tmp}/hf2q-guide-shell.XXXXXX")"
trap 'rm -f "$syntax_file"' EXIT
awk '
    /^```bash$/ { in_bash = 1; next }
    in_bash && /^```$/ { in_bash = 0; print ""; next }
    in_bash { print }
' "$guide" > "$syntax_file"
bash -n "$syntax_file" || fail "a bash code block does not parse"

echo "getting-started guide check passed"
