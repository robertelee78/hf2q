#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
guide="$repo_root/docs/getting-started.md"

fail() {
    echo "getting-started guide check failed: $*" >&2
    exit 1
}

require_literal() {
    local literal="$1"
    grep -Fq -- "$literal" "$guide" || fail "missing '$literal'"
}

reject_literal() {
    local literal="$1"
    if grep -Fq -- "$literal" "$guide"; then
        fail "stale or out-of-scope '$literal' remains"
    fi
}

require_literal "curl -fsSL https://hf2q.us/install.sh | sh"
require_literal "hf2q --version"
require_literal "requires hf2q 0.1.8 or newer"
require_literal "hf2q setup"
require_literal "hf2q setup --accept-defaults"
require_literal "hf2q doctor"
require_literal "hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT"
require_literal "--revision 08c2f075b43bc06456382db6b918a3dcabdcf4dd"
require_literal "--quant q4_k_m"
require_literal "hf2q serve --model \"\$MODEL\""
require_literal "hf2q chat --url http://127.0.0.1:8081/v1"
require_literal "http://127.0.0.1:8081/v1/models"
require_literal "http://127.0.0.1:8081/v1/chat/completions"
require_literal "server later with Ctrl-C"
require_literal "hf2q update --check"
require_literal "hf2q uninstall --yes"

reject_literal "qwen38-abliterated-sft-q5_k_m.gguf"
reject_literal "npm install -"
reject_literal "brew install"
reject_literal "ak setup"
reject_literal "nohup"
reject_literal "pkill -f"
reject_literal "docker run"

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
