#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="${0##*/}"

DEFAULT_OUTPUT="tests/fixtures/ppl-corpus/wikitext2-full.tokens"
DEFAULT_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/hf2q/wikitext2"

# Source choice: Stephen Merity/Salesforce wikitext-2-raw-v1 zip,
# mirrored by ggml-org/ci for llama.cpp CI. The SHA-256 below matches
# the Salesforce/HuggingFace dataset_infos checksum for the original
# S3 artifact:
# https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
WIKITEXT2_URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT2_SHA256="ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
WIKITEXT2_ZIP_NAME="wikitext-2-raw-v1.zip"
WIKITEXT2_TEST_MEMBER="wikitext-2-raw/wiki.test.raw"

MIN_TOKENS=280000
MIN_BYTES=1048576

die() {
    printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
usage: $SCRIPT_NAME [options]

Downloads wikitext-2-raw-v1 test split, tokenizes it with llama-tokenize,
and writes raw little-endian u32 token IDs (no header).

Options:
  --tokenizer-gguf <path>  tokenizer/model GGUF for llama-tokenize
                           (default: HF2Q_QWEN_VOCAB_GGUF)
  --output <path>          output .tokens path
                           (default: $DEFAULT_OUTPUT)
  --cache-dir <path>       cache directory
                           (default: $DEFAULT_CACHE_DIR)
  --force                  rebuild even when output/cache files exist
  --help                   show this help

Source URL:
  $WIKITEXT2_URL

Locked SHA-256:
  $WIKITEXT2_SHA256
EOF
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die "missing sha256sum or shasum"
    fi
}

file_size_bytes() {
    if stat -c %s "$1" >/dev/null 2>&1; then
        stat -c %s "$1"
    else
        stat -f %z "$1"
    fi
}

validate_tokens_file() {
    local path="$1"
    local size count

    [[ -f "$path" ]] || return 1
    size="$(file_size_bytes "$path")" || return 1
    [[ "$size" =~ ^[0-9]+$ ]] || return 1
    (( size >= MIN_BYTES )) || return 1
    (( size % 4 == 0 )) || return 1
    count=$(( size / 4 ))
    (( count >= MIN_TOKENS )) || return 1
    printf '%s %s\n' "$count" "$size"
}

tokenizer_gguf="${HF2Q_QWEN_VOCAB_GGUF:-}"
output="$DEFAULT_OUTPUT"
cache_dir="$DEFAULT_CACHE_DIR"
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokenizer-gguf)
            [[ $# -ge 2 ]] || die "--tokenizer-gguf requires a path"
            tokenizer_gguf="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || die "--output requires a path"
            output="$2"
            shift 2
            ;;
        --cache-dir)
            [[ $# -ge 2 ]] || die "--cache-dir requires a path"
            cache_dir="$2"
            shift 2
            ;;
        --force)
            force=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

output_parent="$(dirname "$output")"
[[ -d "$output_parent" ]] || die "--output parent directory does not exist: $output_parent"

if [[ "$force" -eq 0 && -f "$output" ]]; then
    if read -r cached_count cached_size < <(validate_tokens_file "$output"); then
        cached_sha="$(sha256_file "$output")"
        printf 'cached, skipping: %s has %s tokens (%s bytes); SHA-256 %s\n' \
            "$output" "$cached_count" "$cached_size" "$cached_sha"
        exit 0
    fi
    die "existing output failed validation: $output (use --force to rebuild after inspecting it)"
fi

require_cmd curl
require_cmd unzip
require_cmd python3
require_cmd awk
require_cmd llama-tokenize

[[ -n "$tokenizer_gguf" ]] || die "missing tokenizer GGUF: pass --tokenizer-gguf <path> or set HF2Q_QWEN_VOCAB_GGUF"
[[ -f "$tokenizer_gguf" ]] || die "tokenizer GGUF does not exist: $tokenizer_gguf"

tokenize_help="$(llama-tokenize --help 2>&1 || true)"
for flag in "--model" "--file" "--no-bos" "--ids" "--log-disable"; do
    if ! grep -q -- "$flag" <<<"$tokenize_help"; then
        die "installed llama-tokenize does not advertise required flag $flag"
    fi
done

mkdir -p "$cache_dir"
zip_path="$cache_dir/$WIKITEXT2_ZIP_NAME"
text_path="$cache_dir/wiki.test.raw"

if [[ "$force" -eq 1 ]]; then
    rm -f "$zip_path" "$text_path"
fi

if [[ ! -f "$zip_path" ]]; then
    tmp_zip="$zip_path.tmp.$$"
    rm -f "$tmp_zip"
    if ! curl -L --fail --retry 3 -o "$tmp_zip" "$WIKITEXT2_URL"; then
        rm -f "$tmp_zip"
        die "download failed: $WIKITEXT2_URL"
    fi
    mv "$tmp_zip" "$zip_path"
fi

actual_sha="$(sha256_file "$zip_path")"
if [[ "$actual_sha" != "$WIKITEXT2_SHA256" ]]; then
    die "SHA-256 mismatch for $zip_path: expected $WIKITEXT2_SHA256, got $actual_sha"
fi

if [[ ! -f "$text_path" || "$zip_path" -nt "$text_path" ]]; then
    tmp_text="$text_path.tmp.$$"
    rm -f "$tmp_text"
    if ! unzip -p "$zip_path" "$WIKITEXT2_TEST_MEMBER" >"$tmp_text"; then
        rm -f "$tmp_text"
        die "failed to extract $WIKITEXT2_TEST_MEMBER from $zip_path"
    fi
    if [[ ! -s "$tmp_text" ]]; then
        rm -f "$tmp_text"
        die "extracted text is empty: $WIKITEXT2_TEST_MEMBER"
    fi
    if ! python3 -c 'import sys; open(sys.argv[1], "rb").read().decode("utf-8")' "$tmp_text"; then
        rm -f "$tmp_text"
        die "extracted text is not valid UTF-8: $WIKITEXT2_TEST_MEMBER"
    fi
    mv "$tmp_text" "$text_path"
fi

tmp_output="$output.tmp.$$"
rm -f "$tmp_output"
if ! token_count="$(
    llama-tokenize \
        --model "$tokenizer_gguf" \
        --file "$text_path" \
        --no-bos \
        --ids \
        --log-disable |
        python3 -c '
import re
import struct
import sys

out_path = sys.argv[1]
data = sys.stdin.read()
if re.search(r"[^0-9,\[\]\s-]", data):
    sys.stderr.write("unexpected non-ID output from llama-tokenize; refusing to parse logs as tokens\n")
    sys.exit(2)

count = 0
with open(out_path, "wb") as out:
    for match in re.finditer(r"-?\d+", data):
        value = int(match.group(0))
        if value < 0 or value > 0xFFFFFFFF:
            sys.stderr.write(f"token ID out of u32 range: {value}\n")
            sys.exit(3)
        out.write(struct.pack("<I", value))
        count += 1

if count == 0:
    sys.stderr.write("llama-tokenize produced zero token IDs\n")
    sys.exit(4)

print(count)
' "$tmp_output"
)"; then
    rm -f "$tmp_output"
    die "tokenization failed"
fi

[[ "$token_count" =~ ^[0-9]+$ ]] || die "internal error: non-numeric token count: $token_count"
if (( token_count < MIN_TOKENS )); then
    rm -f "$tmp_output"
    die "token-count validation failed: got $token_count, expected at least $MIN_TOKENS"
fi

output_size="$(file_size_bytes "$tmp_output")"
if (( output_size < MIN_BYTES )); then
    rm -f "$tmp_output"
    die "file-size validation failed: got $output_size bytes, expected at least $MIN_BYTES"
fi
if (( output_size % 4 != 0 )); then
    rm -f "$tmp_output"
    die "file-size validation failed: $output_size is not divisible by 4"
fi
if (( output_size / 4 != token_count )); then
    rm -f "$tmp_output"
    die "token-count validation failed: stdout count $token_count but file holds $(( output_size / 4 )) u32s"
fi

mv "$tmp_output" "$output"
output_sha="$(sha256_file "$output")"
printf 'OK wrote %s tokens (%s bytes) to %s; SHA-256 %s\n' \
    "$token_count" "$output_size" "$output" "$output_sha"
